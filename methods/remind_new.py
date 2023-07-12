
# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from collections import defaultdict
import faiss
import numpy as np
import pandas as pd
import random
import math
import sys
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.data_loader import get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.data_loader import ImageDataset, get_train_baseinit_datalist, MultiProcessLoader, cutmix_data

from methods.cl_manager import CLManagerBase, MemoryBase
import torchvision.models as models
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")
from torchvision import transforms

class REMIND(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        self.dataset = kwargs["dataset"]
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.init_cls = kwargs["init_cls"]
        self.rnd_seed = kwargs['rnd_seed']
        
        self.n_codebooks = kwargs['n_codebooks']
        self.codebook_size = kwargs['codebook_size']
        self.mixup_alpha = kwargs['mixup_alpha']
        self.baseinit_nclasses = kwargs['baseinit_nclasses']
        self.baseinit_epochs = kwargs['baseinit_epochs']
        self.spatial_feat_dim = kwargs['spatial_feat_dim']
        self.baseinit_datalist, self.baseinit_classnames = get_train_baseinit_datalist(self.dataset, self.sigma, self.repeat, self.init_cls, self.rnd_seed)
        
        self.random_resize_crop = RandomResizeCrop(7, scale=(2 / 7, 1.0))

        super().__init__(train_datalist, test_datalist, device, **kwargs)

    def initialize_future(self):
    
        self.memory = REMINDMemory(self.memory_size, self.device)
        print("Begin Base_initialization")
        self.base_initialize()
        print("FINISH BASE INITIALIZATION")
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        
        #self.memory = MemoryBase(self.memory_size)
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.temp_future_batch_idx = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learning_class = self.num_learned_class+1
        self.seen = 0
        self.future_sample_num = len(self.baseinit_datalist)
        self.future_sampling = True
        self.future_retrieval = True

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch)
            self.waiting_batch_idx.append(self.temp_future_batch_idx)

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
        self.temp_future_batch.append(sample)
        self.temp_future_batch_idx.append(self.future_sample_num)
        self.future_num_updates += self.online_iter
        self.future_sample_num += 1
        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(int(self.future_num_updates))
            self.temp_future_batch = []
            self.temp_future_batch_idx = []
            self.future_num_updates -= int(self.future_num_updates)

        return 0
        
    
    def pretrain(self):
        
        for i in range(self.baseinit_nclasses):
            clss = list(self.baseinit_classnames.keys())
            self.cls_dict[clss[i]] = len(self.exposed_classes)
            self.exposed_classes.append(clss[i])
            self.num_learned_class = len(self.exposed_classes)
            self.memory.add_new_class(clss[i])
        print("EXPOSED CLASSES", self.exposed_classes)
        
        self.num_channels = self.model.fc.in_features
        self.baseinit_train_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.RandomCrop(size=(224, 224), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=transforms.InterpolationMode.NEAREST, fill=None),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.baseinit_nclasses).to(self.device)
        self.model.to(self.device)
        self.optimizer = select_optimizer(self.opt_name, 0.001, self.model)
        train_df = pd.DataFrame(self.baseinit_datalist)
        train_dataset = ImageDataset(train_df, self.dataset, self.baseinit_train_transform, cls_list=self.exposed_classes)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_df = pd.DataFrame(self.test_datalist)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(exp_test_df, self.dataset, self.test_transform, cls_list=self.exposed_classes)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        print("Begin Base Initialization!!!")
        for epoch in range(self.baseinit_epochs):
            print("Epoch", epoch)
            self.model.train()
            correct_train = 0
            for data in train_loader:
                inputs = data['image']
                targets = data['label']
                batchs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(batchs)
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
                correct_train += torch.sum(preds == targets.data)

            self.model.eval()
            y_pred = []

            correct_test = 0
            with torch.no_grad():

                for data in test_loader:
                    inputs = data['image']
                    targets = data['label']
                    batchs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(batchs)
                    loss = self.criterion(outputs, targets)
                    y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

                    _, preds = torch.max(outputs, 1)
                    correct_test += torch.sum(preds == targets.data)
                                
            train_acc = 100 * float(correct_train) / len(train_dataset) 
            test_acc = 100 * float(correct_test) / len(test_dataset) 
            print(train_acc, test_acc)
            
        # torch.save(self.model.state_dict(), f"{self.dataset}_cls{self.baseinit_nclasses}_REMIND_sigma{self.sigma}_seed{self.rnd_seed}_pretrained_epoch40.pt")
        
        return train_acc, test_acc
    
    def safe_load_dict(self, model, new_model_state, should_resume_all_params=False):
        old_model_state = model.state_dict()
        c = 0
        if should_resume_all_params:
            for old_name, old_param in old_model_state.items():
                assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                    old_name)
        for name, param in new_model_state.items():
            n = name.split('.')
            beg = n[0]
            end = n[1:]
            if beg == 'module':
                name = '.'.join(end)
            if name not in old_model_state:
                # print('%s not found in old model.' % name)
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            c += 1
            if old_model_state[name].shape != param.shape:
                print('Shape mismatch...ignoring %s' % name)
                continue
            else:
                old_model_state[name].copy_(param)
        if c == 0:
            raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')
        return model
    
    def base_initialize(self):
        train_acc, test_acc = self.pretrain()
        print('Training accuracy: {:.2f}%'.format(float(train_acc)))
        print('Test accuracy: {:.2f}%\n'.format(float(test_acc)))
        
        self.model_G = select_model(self.model_name, self.dataset, 1, G=True).to(self.device)
        self.model_F = select_model(self.model_name, self.dataset, 1, F=True).to(self.device)
        self.model_G.fc = nn.Linear(self.model.fc.in_features, self.baseinit_nclasses).to(self.device)
        self.model_F.fc = nn.Linear(self.model.fc.in_features, self.baseinit_nclasses).to(self.device)
        self.model_G = self.safe_load_dict(self.model_G, self.model.state_dict(), True)
        self.model_F = self.safe_load_dict(self.model_F, self.model.state_dict(), True)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model_F)
      
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        
        # self.lr_scheduler_per_class = {}
        # self.lr_per_class = {}
        # for class_ix in range(0, self.n_classes):
        #     scheduler = select_scheduler(self.sched_name, self.optimizer)
        #     # scheduler = scheduler.load_state_dict(torch.load('cifar100_cls10_REMIND_seed1.pt'))
        #     self.lr_scheduler_per_class[class_ix] = scheduler
        #     self.lr_per_class[class_ix] = self.lr
        
    
        self.model_G.eval()
        train_df = pd.DataFrame(self.baseinit_datalist)
        train_dataset = ImageDataset(train_df, self.dataset, self.baseinit_train_transform, cls_list=self.exposed_classes, data_dir=self.data_dir)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        start_ix = 0
        features_data = np.empty((len(train_loader.dataset), self.num_channels, self.spatial_feat_dim, self.spatial_feat_dim), dtype=np.float32)
        labels_data = np.empty((len(train_loader.dataset), 1), dtype=int)
        item_ixs_data = np.empty((len(train_loader.dataset), 1), dtype=int)
        
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                inputs = data['image']
                targets = data['label']
                batchs = inputs.to(self.device)
                batchs = batchs.float()
                output = self.model_G(batchs)
                end_ix = start_ix + len(output)
                features_data[start_ix:end_ix] = output.detach().cpu().numpy()
                labels_data[start_ix:end_ix] = np.atleast_2d(targets.numpy().astype(int)).transpose()
                item_ixs_data[start_ix:end_ix] = np.atleast_2d(range(start_ix,end_ix)).transpose()
                start_ix = end_ix
        self.fit_pq(features_data, labels_data, item_ixs_data, self.num_channels, self.spatial_feat_dim, self.n_codebooks, self.codebook_size)
        
    # fit quantization model & store data
    def fit_pq(self, feats_base_init, labels_base_init, item_ix_base_init, num_channels, spatial_feat_dim, num_codebooks, codebook_size, batch_size=128):
        train_data_base_init = np.transpose(feats_base_init, (0, 2, 3, 1))
        train_data_base_init = np.reshape(train_data_base_init, (-1, num_channels))
        print("TRAIN", train_data_base_init)
        num_samples = len(labels_base_init)
        # Train PQ
        nbits = int(np.log2(codebook_size))
        self.pq = faiss.ProductQuantizer(num_channels, num_codebooks, nbits)
        self.pq.train(train_data_base_init)
        print("TRAINDONE")
        del train_data_base_init
        
        for i in range(0, num_samples, self.batch_size):
            start = i
            end = min(start + self.batch_size, num_samples)
            data_batch = feats_base_init[start:end]
            batch_labels = labels_base_init[start:end]
            batch_item_ixs = item_ix_base_init[start:end]
            data_batch = np.transpose(data_batch, (0, 2, 3, 1))
            data_batch = np.reshape(data_batch, (-1, num_channels))
            codes = self.pq.compute_codes(data_batch)
            codes = np.reshape(codes, (-1, spatial_feat_dim, spatial_feat_dim, num_codebooks))
            for i in range(len(codes)):
                self.update_completed_memory(codes[i], batch_labels[i][0], batch_item_ixs[i][0])
        print("BASEINIT MEMORY UPDATE DONE")
        
    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model_F.fc.weight.data)
        prev_bias = copy.deepcopy(self.model_F.fc.bias.data)
        self.model_F.fc = nn.Linear(self.model_F.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model_F.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model_F.fc.bias[:self.num_learned_class - 1] = prev_bias
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model_F.fc.parameters()})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
            
    def mixup_data(self, x1, y1, x2, y2, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_x = lam * x1 + (1 - lam) * x2
        y_a, y_b = y1, y2
        return mixed_x, y_a, y_b, lam
    
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        # ongoing_class = None
        for i in range(iterations):
            self.model_F.train()
            self.model_G.eval()
            # self.model.eval()
            data = self.get_batch()
            stream_x = data["image"].to(self.device)
            stream_y = data["label"].to(self.device)
            input_idxs = data["sample_nums"].to(self.device)
            stream_data_batch = self.model_G(stream_x).detach().cpu().numpy()
            stream_data_batch = np.transpose(stream_data_batch, (0, 2, 3, 1))
            stream_data_batch = np.reshape(stream_data_batch, (-1, self.num_channels))
            stream_codes = self.pq.compute_codes(stream_data_batch)
            stream_codes = np.reshape(stream_codes, (-1, self.spatial_feat_dim, self.spatial_feat_dim, self.n_codebooks))
            for x, y, item_ix in zip(stream_codes, stream_y, input_idxs):
                # if (ongoing_class is None or ongoing_class != y):
                #     ongoing_class = y
                #     self.optimizer.param_groups[0]['lr'] = self.lr_per_class[int(y)]
                    
                memory_data = self.memory.retrieval(self.memory_batch_size)
                data_codes = np.empty(
                            ((1 + self.memory_batch_size), self.spatial_feat_dim, self.spatial_feat_dim, self.n_codebooks),
                            dtype=np.uint8)
                data_labels = torch.empty((1 + self.memory_batch_size), dtype=torch.long).to(self.device)
                
                data_codes[0] = x
                data_labels[0] = y.long()
                
                for i in range(len(memory_data)):
                    data_codes[1+i] = memory_data[i]['latent_dict'][0]
                    data_labels[1+i] = torch.tensor(memory_data[i]['latent_dict'][1])
                    
                # Reconstruct
                data_codes = np.reshape(
                    data_codes, ((1 + self.memory_batch_size) * self.spatial_feat_dim * self.spatial_feat_dim, self.n_codebooks))
                data_batch_reconstructed = self.pq.decode(data_codes)
                data_batch_reconstructed = np.reshape(
                    data_batch_reconstructed,(-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels))
                data_batch_reconstructed = torch.from_numpy(
                    np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).to(self.device)
                
                # resize crop augmentation
                # transform_data_batch = torch.empty_like(data_batch_reconstructed)
                # for tens_ix, tens in enumerate(data_batch_reconstructed):
                #     transform_data_batch[tens_ix] = self.random_resize_crop(tens)
                # data_batch_reconstructed = transform_data_batch
                
                # mixup
                # x_prev_mixed, prev_labels_a, prev_labels_b, lam = self.mixup_data(
                #     data_batch_reconstructed[1:1 + self.memory_batch_size],
                #     data_labels[1:1 + self.memory_batch_size],
                #     data_batch_reconstructed[1 + self.memory_batch_size:],
                #     data_labels[1 + self.memory_batch_size:],
                #     alpha=self.mixup_alpha)
                
                # data = torch.empty((1+self.memory_batch_size, self.num_channels, self.spatial_feat_dim, self.spatial_feat_dim))
                # labels_a = torch.zeros(self.memory_batch_size + 1).long()
                # labels_b = torch.zeros(self.memory_batch_size + 1).long()
                # data[0] = data_batch_reconstructed[0]
                # labels_a[0] = y.squeeze()
                # labels_b[0] = y.squeeze()
                # data[1:] = x_prev_mixed.clone()
                # labels_a[1:] = prev_labels_a
                # labels_b[1:] = prev_labels_b
                
                # fit on replay and new sample
                self.optimizer.zero_grad()
                # data = data.to(self.device)
                output = self.model_F(data_batch_reconstructed)
                loss = self.criterion(output, data_labels)
                # output, loss = self.model_forward(data_batch_reconstructed, data_labels, input_idxs)
                # loss = self.mixup_criterion(self.criterion, output, labels_a.to(self.device), labels_b.to(self.device), lam)
                _, preds = output.topk(self.topk, 1, True, True)
                preds = preds.detach().cpu()
                data_labels = data_labels.detach().cpu()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                # loss.backward()
                # self.optimizer.step()

                self.after_model_update()
                
                total_loss += loss.item()
                correct += torch.sum(preds == data_labels.unsqueeze(1)).item()
                num_data += data_labels.size(0)
                # correct += torch.sum(preds == labels_a.unsqueeze(1)).item()
                # correct += torch.sum(preds == labels_b.unsqueeze(1)).item()
                # num_data += labels_a.size(0)
                # num_data += labels_b.size(0)
                # if self.lr_scheduler_per_class is not None:
                #     self.lr_scheduler_per_class[int(y)].step()
                #     self.lr_per_class[int(y)] = self.optimizer.param_groups[0]['lr']
                
                item_ix = int(item_ix.cpu().numpy())
                y = y.cpu().numpy()
                
                self.update_completed_memory(x,y,item_ix)
        return total_loss / (iterations*self.temp_batch_size), correct / num_data

    def update_completed_memory(self, code, label, item_ix):
        item_ix = str(item_ix)
        self.memory.replace_completed_sample(code, label, item_ix)
        

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a.squeeze()) + (1 - lam) * criterion(pred, y_b.squeeze())

    
    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        feature_dict = {}
        label = []
        
        probas = torch.zeros((len(test_loader.dataset), len(self.exposed_classes)))
        all_lbls = torch.zeros((len(test_loader.dataset)))
        start_ix = 0
        with torch.no_grad():
            self.model_G.to(self.device)
            self.model_F.to(self.device)
            self.model_G.eval()
            self.model_F.eval()
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device).float()
                y = y.to(self.device)
              
                data_batch = self.model_G(x).detach().cpu().numpy()
                data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = self.pq.compute_codes(data_batch)
                data_batch_reconstructed = self.pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                      (-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).to(self.device)

                logit = self.model_F(data_batch_reconstructed)
                
                end_ix = start_ix + len(x)
                probas[start_ix:end_ix] = F.softmax(logit.data, dim=1)
                all_lbls[start_ix:end_ix] = y.squeeze()
                start_ix = end_ix
                loss = criterion(logit, y)
                

                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.cpu()
                num_data_l += xlabel_cnt.cpu()
                total_loss += loss.item()
                label += y.tolist()
        
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader.dataset)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model_F)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
    
    def get_forgetting(self, sample_num, test_list, cls_dict, batch_size, n_worker):
        test_df = pd.DataFrame(test_list)
        test_dataset = ImageDataset(
            test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=list(cls_dict.keys()),
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        preds = []
        gts = []
        self.model_G.eval()
        self.model_F.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                logit = self.model_G(x).detach().cpu().numpy
                data_batch = np.transpose(logit.detach().cpu().numpy(), (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = self.pq.compute_codes(data_batch)
                data_batch_reconstructed = self.pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                      (-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).to(self.device)
                logit = self.model_F(data_batch_reconstructed)
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred.cpu().numpy())
                gts.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        if self.gt_label is None:
            gts = np.concatenate(gts)
            self.gt_label = gts
        self.test_records.append(preds)
        self.n_model_cls.append(copy.deepcopy(self.num_learned_class))

class REMINDMemory(MemoryBase):
    def __init__(self, memory_size, device, ood_strategy=None):
        self.latent_dict = {}
        self.rehearsal_ixs = []

        super().__init__(memory_size, device, ood_strategy=None)

    def __len__(self):
        return len(self.rehearsal_ixs)

    def replace_completed_sample(self, code, label, item_ix):
            
        if len(self.rehearsal_ixs) < self.memory_size:
            if item_ix not in list(self.latent_dict.keys()):
                self.cls_idx[label].append(item_ix)
                self.rehearsal_ixs.append(item_ix)
                self.cls_count[label] += 1
            self.latent_dict[item_ix] = [code, label]
        else:
            if item_ix in list(self.latent_dict.keys()):
                self.latent_dict[item_ix] = [code, label]
            else:
                label_ind = self.cls_count.index(max(self.cls_count))
                cls_ind = np.random.randint(0, len(self.cls_idx[label_ind]))
                ind = self.rehearsal_ixs.index(self.cls_idx[label_ind][cls_ind])
                print("GOODBYE", self.cls_idx[label_ind][cls_ind])
                self.cls_count[label_ind] -= 1
                del self.latent_dict[self.cls_idx[label_ind][cls_ind]]
                del self.cls_idx[label_ind][cls_ind]
                self.latent_dict[item_ix] = [code, label]
                self.rehearsal_ixs[ind] = item_ix
                self.cls_idx[label].append(item_ix)
                self.cls_count[label] += 1

    
    def whole_retrieval(self):
        memory_batch = []
        indices = list(range(len(self.rehearsal_ixs)))
        for i in indices:
            data = {}
            data['latent_dict'] = self.latent_dict[i]
            data['ix'] = self.rehearsal_ixs[i]
            memory_batch.append(data)
        return memory_batch

    def retrieval(self, size):
        memory_batch = []
        indices = np.random.choice(range(len(self.rehearsal_ixs)), size=size, replace=False)
        for i in indices:
            data = {}
            data['ix'] = self.rehearsal_ixs[i]
            data['latent_dict'] = self.latent_dict[self.rehearsal_ixs[i]]
            memory_batch.append(data)
        return memory_batch
    
class RandomResizeCrop(object):
    """Randomly crops tensor then resizes uniformly between given bounds
    Args:
        size (sequence): Bounds of desired output sizes.
        scale (sequence): Range of size of the origin size cropped
        ratio (sequence): Range of aspect ratio of the origin aspect ratio cropped
        interpolation (int, optional): Desired interpolation. Default is 'bilinear'
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        #        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (3-d tensor (C,H,W)): Tensor to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size(1) * img.size(2)

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size(1) and h <= img.size(2):
                i = random.randint(0, img.size(2) - h)
                j = random.randint(0, img.size(1) - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size(1) / img.size(2)
        if (in_ratio < min(ratio)):
            w = img.size(1)
            h = int(w / min(ratio))
        elif (in_ratio > max(ratio)):
            h = img.size(2)
            w = int(h * max(ratio))
        else:  # whole image
            w = img.size(1)
            h = img.size(2)
        i = int((img.size(2) - h) // 2)
        j = int((img.size(1) - w) // 2)
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (3-D tensor (C,H,W)): Tensor to be cropped and resized.
        Returns:
            Tensor: Randomly cropped and resized Tensor.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = img[:, i:i + h, j:j + w]  ##crop
        return torch.nn.functional.interpolate(img.unsqueeze(0), self.size, mode=self.interpolation,
                                               align_corners=False).squeeze(0)

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, scale={1}, ratio={2}, interpolation={3})'.format(self.size,
                                                                                                      self.scale,
                                                                                                      self.ratio,
                                                                                                      interpolate_str)
