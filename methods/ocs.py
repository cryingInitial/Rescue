import logging
import torch
import torch.nn as nn
import copy
import os
import time
import PIL
import numpy as np
import datetime
import random
from methods.cl_manager import MemoryBase
from methods.er_new import ER

from utils.data_loader import MultiProcessLoader
from utils.augment import Preprocess, get_statistics
from utils.autograd_hacks import add_hooks, remove_hooks, compute_grad1, clear_backprops
import torchvision.transforms as T


from torch.utils.tensorboard import SummaryWriter


# [INFO] You can find important statements by searching "MILESTONE" on the log file.
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class OCS(ER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        self.n_tasks = kwargs["n_tasks"]
        print("n_tasks", self.n_tasks)
        self.total_samples = len(train_datalist)        
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.base = 0
        self.update_count = 0
        
    def initialize_future(self):
        self.class_per_task = self.n_classes // self.n_tasks
        print(f"Class per task: {self.class_per_task}")
        print("TEMP BATCH SIZE:", self.temp_batch_size)
        print("MEMORY BATCH SIZE:", self.memory_batch_size)
        self.samples_per_task = len(self.train_datalist) // self.n_tasks
        
        self.data_stream = iter(self.train_datalist)
        
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = OCSMemory(self.memory_size, self.device, self.sigma, self.class_per_task)
        # self.memory = MemoryBase(self.memory_size, self.device, self.sigma)
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.temp_future_batch2 = []
        self.temp_future_batch_idx2 = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True
        self.task_num = 0
        self.candidates = []
        
        # ocs 2X batch_size, for random sampling
        self.temp_batch_size = self.temp_batch_size * 2
        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        
        for i in range(self.future_steps):
            self.load_batch()

    def online_before_task(self, task):
        # task 변경 적용
        self.task_num = task
        self.memory.task_change(self.task_num)
        
        # coreset을 기반으로 memory update
        # if task > 0:
        #     self.memory.update_coreset(self.model, self.candidates, self.train_datalist, data_dir=self.data_dir)
        #     self.candidates = []
        

    def online_train(self, iterations=1):
                              
        # iterations의 절반은 random sampling, 절반은 coreset sampling + memory update                                                                                                                                                                                            
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        for i in range(iterations):

            self.update_count += 1
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            sample_nums = data['sample_nums'].to(self.device)
            
            x_stream = x[:self.temp_batch_size]
            y_stream = y[:self.temp_batch_size]
            x_memory = x[self.temp_batch_size:]
            y_memory = y[self.temp_batch_size:]
            sample_nums_stream = sample_nums[:self.temp_batch_size]
            sample_nums_memory = sample_nums[self.temp_batch_size:]
            self.before_model_update()
            
            if i < iterations // 2:
                print("RS", end=" ")
                pick = torch.randperm(len(x_stream))[:self.temp_batch_size // 2]
                
            else:
                print("CS", end=" ")
                _eg = self.compute_and_flatten_example_grads(self.model, self.optimizer, x_stream, y_stream, self.task_num, is_total=True)
                _g = torch.mean(_eg, dim=0)
                ref_grads = None
                
                if len(self.exposed_classes) > self.class_per_task:
                    is_included = []
                    for y_memory_ in y_memory:
                        if y_memory_ < len(self.exposed_classes) - self.class_per_task:
                            is_included.append(True)
                        else:
                            is_included.append(False)
                    is_included = torch.tensor(is_included).to(self.device)
                    x_memory_for_affinity = x_memory[is_included]
                    y_memory_for_affinity = y_memory[is_included]
                    if len(x_memory_for_affinity) != 0:
                        print("affinity", len(x_memory_for_affinity), end=" ")
                        ref_grads = self.compute_and_flatten_example_grads(self.model, self.optimizer, x_memory_for_affinity, y_memory_for_affinity, self.task_num, is_total=False)
                    
                pick = self.sample_selection(_g, _eg, ref_grads)[:self.temp_batch_size // 2]

                _eg = _eg.detach().cpu()
                _g = _g.detach().cpu()
                if ref_grads is not None:
                    ref_grads = ref_grads.detach().cpu()
                
                if i == iterations - 1:
                    print("MU", end=" ")
                    sample_ids = [self.train_datalist[i] for i in pick + self.base]
                    self.memory.update_memory_with_ocs(sample_ids)
                    
            x_stream = x_stream[pick]
            y_stream = y_stream[pick]
            self.optimizer.zero_grad()
            
            # important stream loss
            logit, loss = self.model_forward(x_stream, y_stream, sample_nums_stream)
            
            # coreset memory loss
            if len(x_memory) > 0:
                logit_memory, loss_memory = self.model_forward(x_memory, y_memory, sample_nums_memory)
                loss += loss_memory
            
            loss = loss.mean()
                
            _, preds = logit.topk(self.topk, 1, True, True)
            
            # update
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y_stream.unsqueeze(1)).item()
            num_data += y_stream.size(0)

        self.base += self.temp_batch_size
        print("# of update:", self.update_count)
        print("# of data:", len(y_stream), len(y_memory))
        
        return total_loss / iterations, correct / num_data
    
    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        ocs_sample_num = sample_num + self.temp_batch_size - 1
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        if ocs_sample_num < len(self.train_datalist) and self.train_datalist[ocs_sample_num]['klass'] not in self.exposed_classes:
            self.add_new_class(self.train_datalist[ocs_sample_num]['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []
            
    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            print("!cls added", sample['klass'])
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            
        self.temp_future_batch.append(sample)
        self.temp_future_batch_idx.append(self.future_sample_num)
        
        self.future_num_updates += self.online_iter
        
        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(int(self.future_num_updates))
            for stored_sample in self.temp_future_batch:
                # self.update_memory(sample, self.future_sample_num)
                self.future_sample_num += 1
            self.temp_future_batch = []
            self.temp_future_batch2 = []
            self.future_num_updates -= int(self.future_num_updates)
        return 0
    
    def generate_waiting_batch(self, iterations):
        
        for i in range(iterations):

            memory_batch, memory_batch_idx = self.memory.retrieval(self.memory_batch_size)
            self.waiting_batch.append(self.temp_future_batch + self.temp_future_batch2 + memory_batch)
            self.waiting_batch_idx.append(self.temp_future_batch_idx + self.temp_future_batch_idx2 + memory_batch_idx)
    
    # From OCS paper, github.com/rahafaljundi/OCS
    def sample_selection(self, g, eg, ref_grads=None, attn=None):
        # grap gpu memory
        g, eg = g.to(self.device), eg.to(self.device)
        if ref_grads is not None: ref_grads = ref_grads.to(self.device)
        
        ng = torch.norm(g)
        neg = torch.norm(eg, dim=1)
        mean_sim = torch.matmul(g,eg.t()) / torch.maximum(ng*neg, torch.ones_like(neg)*1e-6)
        negd = torch.unsqueeze(neg, 1)

        cross_div = torch.matmul(eg,eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd)*1e-6)
        mean_div = torch.mean(cross_div, 0)

        coreset_aff = torch.tensor(0.).to(self.device)
        if ref_grads is not None:
            ref_ng = torch.norm(ref_grads)
            coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng*neg, torch.ones_like(neg)*1e-6)

        measure = mean_sim - mean_div + 1000 * coreset_aff # tau = 1000
        _, u_idx = torch.sort(measure, descending=True)
        
        # release all gpu memory
        del g, eg, neg, mean_sim, negd, cross_div, mean_div, coreset_aff, measure
        return u_idx.cpu().numpy()
    
    def flatten_grads(self, grads):
        return torch.cat([grad.view(-1) for grad in grads])
    def compute_and_flatten_example_grads(self, m, criterion, data, target, task_id, is_total=False):
        _eg = []
        criterion2 = nn.CrossEntropyLoss().to(self.device)
        # m.eval()
        m.zero_grad()
        
        add_hooks(m)
        pred = m(data)
        criterion2(pred, target).backward(retain_graph=True)
        compute_grad1(m)
        
        clear_backprops(m)
        remove_hooks(m)

        total_grads = None
        avg_grads = None
        
        for param in m.parameters():
            try:
                if total_grads == None:
                    total_grads = param.grad1.detach().view(len(data), -1)
                    avg_grads = param.grad.detach().view(-1)
                else:
                    total_grads = torch.cat([total_grads, param.grad1.detach().view(len(data), -1)], -1)
                    avg_grads = torch.cat([avg_grads, param.grad.detach().view(-1)])
            except:
                pass
            
        if is_total:
            del avg_grads
            return total_grads
        else:
            del total_grads
            return avg_grads   
    
class OCSMemory(MemoryBase):
    
    def __init__(self, memory_size, device, sigma, class_per_task):
        super().__init__(memory_size, device)
        self.task = 0
        self.task_idx = [[]]
        self.task_count = []
        self.sample_per_class = memory_size
        self.sample_per_task = memory_size
        self.device = device
        self.sigma = sigma
        self.seen = 0
        self.class_per_task = class_per_task
        
    def update_memory_with_ocs(self, samples):
        self.class_balanced_replace_samples(samples)
     
    
    def class_balanced_replace_samples(self, samples):
        for sample in samples:
            cls_of_sample = self.cls_dict[sample['klass']]
            if self.memory_size > len(self.images):
                self.cls_idx[cls_of_sample].append(len(self.images))
                self.images.append(sample)
                self.labels.append(cls_of_sample)
            else:
                max_value = np.max([len(self.cls_idx[i]) for i in range(len(self.cls_idx))])
                idxes = np.where(np.array([len(self.cls_idx[i]) for i in range(len(self.cls_idx))]) == max_value)[0]
                idx = idxes[random.randint(0, len(idxes)-1)]
                switch_idx = self.cls_idx[idx].pop(random.randint(0, len(self.cls_idx[idx])-1))
                self.cls_idx[cls_of_sample].append(switch_idx)
                self.images[switch_idx] = sample
                self.labels[switch_idx] = cls_of_sample

    def task_change(self, task_id):
        self.task = task_id
    
    def retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        memory_batch_idx = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        
        for i in indices:
            memory_batch.append(self.images[i])
            memory_batch_idx.append(0)
                
        return memory_batch, memory_batch_idx

