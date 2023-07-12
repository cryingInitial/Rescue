# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from methods.cl_manager import CLManagerBase
from utils.train_utils import DR_loss, Accuracy, DR_Reverse_loss
import torch
import torch.nn as nn
import copy
from collections import defaultdict
from methods.etf_er_resmem_ver3 import ETF_ER_RESMEM_VER3
import torch.nn.functional as F
import pickle5 as pickle
from utils.data_loader import ImageDataset, MultiProcessLoader, cutmix_data, get_statistics, generate_new_data, generate_masking
import math
import utils.train_utils 
import os
from utils.data_worker import load_data, load_batch
from utils.train_utils import select_optimizer, select_moco_model, select_scheduler, SupConLoss
from utils.augment import my_segmentation_transforms
logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class ETF_ER_RESMEM_VER8(ETF_ER_RESMEM_VER3):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        
        self.selfsup_temp = kwargs["selfsup_temp"]
        self.selfsup_criterion = SupConLoss(temperature=self.selfsup_temp).to(self.device)
        self.moco_criterion = nn.CrossEntropyLoss().to(self.device)
        self.compute_accuracy = Accuracy(topk=self.topk)

        # MOCO parameters
        self.moco_k = kwargs["moco_k"]
        self.moco_dim = kwargs["moco_dim"]
        self.moco_T = kwargs["moco_T"]
        self.moco_m = kwargs["moco_m"]
        self.moco_coeff = kwargs["moco_coeff"]
        
        self.image_dict = defaultdict(list)
        self.label_dict = defaultdict(list)

        self.use_neck_forward = kwargs["use_neck_forward"]
        self.model = select_moco_model(self.model_name, self.dataset, 1, pre_trained=False, Neck=self.use_neck_forward, K=self.moco_k, dim=self.moco_dim).to(self.device)
        
        # moco initialize
        self.ema_model = copy.deepcopy(self.model) #select_moco_model(self.model_name, self.dataset, 1, pre_trained=False, Neck=self.use_neck_forward).to(self.device)
        for param_q, param_k in zip(
            self.model.parameters(), self.ema_model.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1.0 - self.moco_m)
    
    
    def model_forward(self, x, y, sample_nums):
        
        with torch.cuda.amp.autocast(self.use_amp):
            x_q = x[:len(x)//2]
            x_k = x[len(x)//2:]
            y = y[:len(y)//2]
            
            target = self.etf_vec[:, y].t()
            proj_output, feature = self.model(x_q)
            feature = self.pre_logits(feature)

            if self.loss_criterion == "DR":
                loss = self.criterion(feature, target)
                residual = (target - feature).detach()
                
            elif self.loss_criterion == "CE":
                logit = feature @ self.etf_vec
                loss = self.criterion(logit, y)
                residual = (target - feature).detach()
            
            if len(proj_output.shape) == 1:    
                proj_output = proj_output.unsqueeze(dim=0)
                                
            ### Moco Loss Calculation ###
            q = nn.functional.normalize(proj_output, dim=1)
            
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                    
                k, _ = self.ema_model(x_k)

                if len(k.shape) == 1:    
                    k = k.unsqueeze(dim=0)

                k = nn.functional.normalize(k, dim=1)

            ##compute logits##
            # positive logits: Nx1
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.model.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.moco_T 

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

            # dequeue and enqueue
            self.model._dequeue_and_enqueue(k)
            selfsup_loss = self.moco_criterion(logits, labels)
            
            # TODO loss balancing
            loss += (self.moco_coeff * selfsup_loss)

            # residual dict update
            if self.use_residual:
                for label in torch.unique(y):
                    index = (y==label).nonzero(as_tuple=True)[0]
                    self.image_dict[label.item()].extend(x_k[index].cpu())
                    self.label_dict[label.item()].extend(y[index].cpu())
                    
                    self.image_dict[label.item()] = self.image_dict[label.item()][-self.residual_num:]
                    self.label_dict[label.item()] = self.label_dict[label.item()][-self.residual_num:]
            
                
            # accuracy calculation
            with torch.no_grad():
                cls_score = feature.detach() @ self.etf_vec
                acc, correct = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                acc = acc.item()
        
        return loss, feature, correct
    
    def update_residual_feature(self):
        image_list = torch.stack(sum([v for v in self.image_dict.values()], []))
        label_list = torch.stack(sum([v for v in self.label_dict.values()], []))
        print("image_list", image_list.shape)
        print("label_list", label_list.shape)
        batch_size = 256
        with torch.no_grad():
            for i in range((len(image_list) // batch_size) + 1):
                print("from", i*batch_size, "to", min((i+1)*batch_size, len(image_list)))
                images = image_list[i*batch_size : min((i+1)*batch_size, len(image_list))]
                labels = label_list[i*batch_size : min((i+1)*batch_size, len(image_list))]
                if len(images)!=0:
                    _, features = self.model(images.to(self.device))
                    features = self.pre_logits(features) # added
                    target = self.etf_vec[:, labels].t()
                    residual = (target - features).detach()

                    # residual dict update
                    if self.use_residual:
                        for label in torch.unique(labels):
                            index = (labels==label).nonzero(as_tuple=True)[0]
                            self.residual_dict[label.item()].extend(residual[index])
                            self.feature_dict[label.item()].extend(features.detach()[index])
                            
                            if len(self.residual_dict[label.item()]) > self.residual_num:
                                self.residual_dict[label.item()] = self.residual_dict[label.item()][-self.residual_num:]
                                self.feature_dict[label.item()] = self.feature_dict[label.item()][-self.residual_num:] 
            
    
    def evaluation(self, test_loader, criterion):
        self.model.eval()
        self.update_residual_feature()
        return super().evaluation(test_loader, criterion)

    def load_batch(self):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            # 2배로 batch를 늘려주기
            self.dataloader.load_batch(self.waiting_batch[0] + self.waiting_batch[0], self.memory.cls_dict, self.waiting_batch_idx[0] + self.waiting_batch_idx[0])
            del self.waiting_batch[0]
            del self.waiting_batch_idx[0]


