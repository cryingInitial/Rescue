import logging
import torch
import torch.nn as nn
import copy
import os
import time
import PIL
import numpy as np
import datetime

from methods.cl_manager import MemoryBase
from methods.er_new import ER

from utils.data_loader import MultiProcessLoader
from utils.augment import Preprocess
from utils.autograd_hacks import add_hooks, remove_hooks, compute_grad1, clear_backprops

from torch.utils.tensorboard import SummaryWriter

# [INFO] You can find important statements by searching "MILESTONE" on the log file.
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class OCS(ER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        self.n_tasks = kwargs["n_tasks"]
        print("n_tasks", self.n_tasks)
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.total_samples = len(train_datalist)
        self.base = 0
        self.ratio = 0
        self.sub_step = self.n_tasks * 2
        self.samples_per_sub_step = self.total_samples // self.sub_step
        
    def initialize_future(self):
        self.class_per_task = self.n_classes // self.n_tasks
        print(f"Class per task: {self.class_per_task}")
        self.samples_per_task = len(self.train_datalist) // self.n_tasks
        self.data_stream = iter(self.train_datalist)
        self.memory_batch_size //= 2
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory_size += (self.temp_batch_size) // 2
        self.memory = OCSMemory(self.memory_size, self.device, self.sigma, self.class_per_task)
        # self.memory = MemoryBase(self.memory_size, self.device, self.sigma)
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
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

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        
        for i in range(self.future_steps):
            self.load_batch()

    def online_before_task(self, task):
        # task 변경 적용
        self.task_num = task
        self.memory.task_change(self.task_num)
        
        # coreset을 기반으로 memory update
        if task > 0:
            self.memory.update_coreset(self.model, self.candidates, self.train_datalist, data_dir=self.data_dir)
            self.candidates = []
        

    def online_train(self, iterations=1):
                                                                                                                                                                                                                          
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        for i in range(iterations // 2):
            # ER과의 공평한 비교를 위해 iterations을 2로 나눠서 ER과 동일한 iteration 수행
            
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
            
            # Picking some samples from stream
            pick = []
            if self.base % self.samples_per_sub_step < self.samples_per_sub_step * self.ratio or \
                self.base > (self.sub_step - 1) * self.samples_per_sub_step: # No coreset selection for last step
                pick = torch.randperm(len(x_stream))[:self.temp_batch_size // 2]
            else:
                print("CORESET SAMPLING")
                _eg = self.compute_and_flatten_example_grads(self.model, self.optimizer, x_stream, y_stream, self.task_num, is_total=True)
                _g = torch.mean(_eg, dim=0)
                ref_grads = None
                if len(x_memory) > 0:
                    # ref_pred = self.model(x_memory)
                    # ref_loss = self.criterion(ref_pred, y_memory)
                    # ref_loss.backward()
                    # ref_grads = copy.deepcopy(self.flatten_grads(self.model))
                    ref_grads = self.compute_and_flatten_example_grads(self.model, self.optimizer, x_memory, y_memory, self.task_num, is_total=False)
                    
                pick = self.sample_selection(_g, _eg, ref_grads)[:self.temp_batch_size // 2]

                _eg = _eg.detach().cpu()
                _g = _g.detach().cpu()
                if ref_grads is not None:
                    ref_grads = ref_grads.detach().cpu()

                if self.base % self.samples_per_task < self.samples_per_sub_step * (1 + self.ratio) and i == (iterations - 1):
                    self.candidates += list(pick + self.base)
                
            x_stream = x_stream[pick]
            y_stream = y_stream[pick]
            self.optimizer.zero_grad()
            
            # important stream loss
            logit, loss = self.model_forward(x_stream, y_stream, sample_nums_stream)
            
            # coreset memory loss
            if len(x_memory) > 0:
                logit_memory, loss_memory = self.model_forward(x_memory, y_memory, sample_nums_memory)
                loss += loss_memory * 0.5 # 0.5 is hyperparameter
                
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
        
        return total_loss / iterations, correct / num_data
    
    def generate_waiting_batch(self, iterations):
        # ER과의 공평한 비교를 위해 iterations을 2로 나눠서 ER과 동일한 iteration 수행
        for i in range(iterations // 2):
            memory_batch, memory_batch_idx = self.memory.retrieval(self.memory_batch_size)
            self.waiting_batch.append(self.temp_future_batch + memory_batch)
            self.waiting_batch_idx.append(self.temp_future_batch_idx + memory_batch_idx)
    
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
    
    def update_memory(self, sample, sample_num):
        # only when OCS is not used
        if self.task_num == 0:
            self.reservoir_memory(sample, sample_num)
            
    def report_training(self, sample_num, train_loss, train_acc):
        sample_num //= 2
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-(sample_num * 2)) / (sample_num * 2)))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc, cls_acc):
        sample_num //= 2        
        self.writer.add_scalar(f"test/loss", avg_loss, sample_num)
        self.writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | TFLOPs {self.total_flops/1000:.2f}"
        )
        for idx in range(self.num_learned_class):
            acc = cls_acc[idx]
            logger.info(
                f"Class_Acc | Sample # {sample_num} | cls{idx} {acc:.4f}"
            )
            
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
    
    def is_full(self):
        return len(self.images) >= self.memory_size
    
    def task_change(self, task):
        if task != 0:
            self.task_idx.append([])
        self.task = task
      
    def add_new_class(self, class_name):
        super().add_new_class(class_name)
    
    def update_coreset(self, model, candidates, train_datalist, candidate_size=2000, data_dir = None):
        
        self.sample_per_task = self.memory_size // self.task
        self.sample_per_class = self.memory_size // (self.class_per_task * self.task)
        
        # er to ocs
        if self.task == 1:
            print("MILESTONE: ER TO OCS")
            # initialize the memory, empty all the er samples
            self.task_idx[0] = []
            self.cls_idx = [[] for _ in range(len(self.cls_list))]
            self.images = []
            self.labels = []
            
        ###########################
        # UPDATE EXISTING CORESET #
        ###########################
        for t in range(self.task - 1):
            assert len(self.task_idx[t]) != 0
            print(f"MILESTONE: UPDATE before {self.task_idx}")
            
            preprocess = Preprocess()
            images = []
            labels = []
            for idx in self.task_idx[t]:
                img_name = self.images[idx]["file_name"]
                img_path = os.path.join(data_dir, img_name)
                image = PIL.Image.open(img_path).convert("RGB")
                images.append(preprocess(image).to(self.device))
                labels.append(self.labels[idx])
            
            images = torch.stack(images)
            labels = torch.Tensor(labels).long().to(self.device)
            pred = model(images)
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(pred, labels)
            loss.backward()
            
            _tid_eg = self.compute_and_flatten_example_grads(model, nn.CrossEntropyLoss(reduction='none'), images, labels, t)
            _tid_g = torch.mean(_tid_eg, 0)
            sorted = self.sample_selection(_tid_g, _tid_eg)
            # 중요도 순으로 정렬
            self.task_idx[t] = [self.task_idx[t][i] for i in sorted]
            
            # CLASS 중요도 순으로 정렬 (DISJOINT)
            if self.sigma == 0:
                for c_t in range(t * self.class_per_task, (t + 1) * self.class_per_task):
                    new_cls_idx_segment = []
                    for idx in self.task_idx[t]:
                        for c_idx in self.cls_idx[c_t]:
                            if c_idx == idx:
                                new_cls_idx_segment.append(idx)
                    self.cls_idx[c_t] = new_cls_idx_segment
                    
            print(f"MILESTONE: UPDATE after {self.task_idx}")
                            
        ###########
        # ADD NEW #
        ###########
        candidates = candidates[:candidate_size]
        pick = torch.randperm(len(candidates))
        candidates = [candidates[i] for i in pick]
        
        # TASK 별로 공평하게 분배 (BLURRY)
        if self.sigma > 0:    
            candidates = candidates[:self.sample_per_task]  
            for c_idx in candidates:
                if self.memory_size > len(self.images):
                    self.task_idx[self.task-1].append(len(self.images))
                    self.images.append(train_datalist[c_idx])
                    self.labels.append(self.cls_dict[train_datalist[c_idx]['klass']])
                else:
                    switch_idx = max(self.task_idx, key=len).pop()
                    self.task_idx[self.task-1].append(switch_idx)
                    self.images[switch_idx] = train_datalist[c_idx]
                    self.labels[switch_idx] = self.cls_dict[train_datalist[c_idx]['klass']]
            
            print("MILESTONE: BLURRY REPORT")
            print('task_idx', [len(self.task_idx[i]) for i in range(len(self.task_idx))])
            
        # CLASS 별로 공평하게 분배 (DISJOINT)
        else:
            for c_idx in candidates:
                candidate = train_datalist[c_idx]
                cls_of_candidate = self.cls_dict[candidate['klass']]
                if self.memory_size > len(self.images):
                    self.task_idx[self.task-1].append(len(self.images))
                    self.cls_idx[cls_of_candidate].append(len(self.images))
                    self.images.append(candidate)
                    self.labels.append(self.cls_dict[candidate['klass']])
                else:
                    print(max(self.cls_idx, key=len))
                    switch_idx = max(self.cls_idx, key=len).pop()
                    switching_sample = self.images[switch_idx]
                    swithcing_sample_task = self.cls_dict[switching_sample['klass']] // self.class_per_task
                    self.task_idx[swithcing_sample_task].pop(self.task_idx[swithcing_sample_task].index(switch_idx))
                    self.task_idx[self.task-1].append(switch_idx)
                    self.cls_idx[cls_of_candidate].append(switch_idx)
                    self.images[switch_idx] = candidate
                    self.labels[switch_idx] = self.cls_dict[candidate['klass']]
        
            print("MILESTONE: DISJOINT REPORT")
            print('task_idx', [len(self.task_idx[i]) for i in range(len(self.task_idx))])
            print('cls_idx', [len(self.cls_idx[i]) for i in range(len(self.cls_idx))])
            
            
        print("MILESTONE: FIANL REPORT")
        print([self.cls_dict[image['klass']] for image in self.images], len(self.images))
    
    
    # def compute_and_flatten_example_grads(self, m, criterion, data, target, task_id):
    #     _eg = []
    #     criterion2 = nn.CrossEntropyLoss().to(self.device)
    #     m.eval()
    #     m.zero_grad()
        
    #     add_hooks(m)
    #     pred = m(data)
    #     criterion2(pred, target).backward(retain_graph=True)
    #     compute_grad1(m)
        
    #     total_grads = None
    #     avg_grads = None
    #     for param in m.parameters():
    #         try:
    #             if total_grads == None:
    #                 total_grads = param.grad1.cpu().detach().view(len(data), -1)
    #                 avg_grads = param.grad.cpu().detach().view(-1)
    #             else:
    #                 total_grads = torch.cat([total_grads, param.grad1.detach().view(len(data), -1)], -1)                    
    #                 avg_grads = torch.cat([avg_grads, param.grad.detach().view(-1)])
    #         except:
    #             pass
                
    #     return total_grads.cpu(), avg_grads.cpu()
            
    def compute_and_flatten_example_grads(self, m, criterion, data, target, task_id):
        _eg = []
        criterion2 = nn.CrossEntropyLoss(reduction='none').to(self.device)
        m.eval()
        m.zero_grad()
        pred = m(data)
        loss = criterion2(pred, target)
        for idx in range(len(data)):
            loss[idx].backward(retain_graph=True)
            _g = self.flatten_grads(m, numpy_output=True)
            _eg.append(torch.Tensor(_g))
            m.zero_grad()
        return torch.stack(_eg)
    
    # Extract gradients from the model
    def flatten_grads(self, m, numpy_output=False, bias=True, only_linear=False):
        total_grads = []
        for name, param in m.named_parameters():
            if only_linear:
                if (bias or not 'bias' in name) and 'linear' in name:
                    total_grads.append(param.grad.detach().view(-1))
            else:
                if (bias or not 'bias' in name) and not 'bn' in name and not 'IC' in name:
                    try:
                        total_grads.append(param.grad.detach().view(-1))
                    except AttributeError:
                        pass
        total_grads = torch.cat(total_grads)
        if numpy_output:
            return total_grads.cpu().detach().numpy()
        return total_grads
    
    def sample_selection(self, g, eg, ref_grads=None, attn=None):
        ng = torch.norm(g)
        neg = torch.norm(eg, dim=1)
        mean_sim = torch.matmul(g,eg.t()) / torch.maximum(ng*neg, torch.ones_like(neg)*1e-6)
        negd = torch.unsqueeze(neg, 1)

        cross_div = torch.matmul(eg,eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd)*1e-6)
        mean_div = torch.mean(cross_div, 0)

        coreset_aff = 0.
        if ref_grads is not None:
            ref_ng = torch.norm(ref_grads)
            coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng*neg, torch.ones_like(neg)*1e-6)

        measure = mean_sim - mean_div + 1000 * coreset_aff # tau = 1000
        _, u_idx = torch.sort(measure, descending=True)
        return u_idx.cpu().numpy()
    
    def replace_sample(self, sample, idx=None, sample_num=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.task_idx[self.task].append(len(self.images))
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
            if sample_num is not None:
                self.sample_nums.append(sample_num)
        else:
            assert idx < self.memory_size
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            for task in range(self.task):
                if idx in self.task_idx[task]:
                    self.task_idx[task].remove(idx)
                    break
            self.task_idx[self.task].remove(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_dict[sample['klass']]
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            self.task_idx[self.task].append(idx)
            if sample_num is not None:
                self.sample_nums[idx] = sample_num
            