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

class ETF_ER_RESMEM_VER7(ETF_ER_RESMEM_VER3):
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
            print("x", x.shape)
            x_q = x[:len(x)//2]
            x_k = x[len(x)//2:]
            y = y[:len(y)//2]
            
            target = self.etf_vec[:, y].t()
            feature, proj_output = self.model(x_q)
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
                    
                _, k = self.ema_model(x_k)

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
                for idx, t in enumerate(y):
                    self.residual_dict[t.item()].append(residual[idx])
                    self.feature_dict[t.item()].append(feature.detach()[idx])
                        
                    if len(self.residual_dict[t.item()]) > self.residual_num:
                        self.residual_dict[t.item()] = self.residual_dict[t.item()][1:]
                        self.feature_dict[t.item()] = self.feature_dict[t.item()][1:]
                
            # accuracy calculation
            with torch.no_grad():
                cls_score = feature.detach() @ self.etf_vec
                acc, correct = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                acc = acc.item()
        
        return loss, feature, correct
    


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

    def evaluation(self, test_loader, criterion):
        print("Memory State")
        print(self.memory.cls_count)
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes).to(self.device)
        num_data_l = torch.zeros(self.n_classes).to(self.device)
        self.model.eval()


        if self.use_residual:
            residual_list = torch.stack(sum([v for v in self.residual_dict.values()], [])) #torch.stack(list(self.residual_dict.values())[0])
            feature_list = torch.stack(sum([v for v in self.feature_dict.values()], [])) #torch.stack(list(self.feature_dict.values())[0])
            print("feature_list", feature_list.shape, "residual_list", residual_list.shape)
            
            # residual dict 내의 feature들이 어느정도 잘 모여있는 상태여야 residual term good
            nc1_feature_dict = defaultdict(list)
            mean_vec_list = defaultdict(list)
            for cls in list(self.feature_dict.keys()):
                stacked_feature_dict = torch.stack(self.feature_dict[cls]).detach()
                nc1_feature_dict[cls] = stacked_feature_dict / torch.norm(stacked_feature_dict, p=2, dim=1, keepdim=True)
                mean_vec_list[cls] = torch.mean(stacked_feature_dict, dim=0)
                
            mu_G = torch.mean(torch.stack(list(self.feature_dict.values())[0]), dim=0)
            whole_cov_value = self.get_within_whole_class_covariance(mu_G, feature_list)
            
            if self.residual_strategy == "within":
                cov_tensor = self.get_within_class_covariance(mean_vec_list, nc1_feature_dict)
                prob = torch.ones_like(cov_tensor).to(self.device) - cov_tensor / whole_cov_value
                print("prob")
                print(prob)
            elif self.residual_strategy == "nc1":
                nc1_tensor = self.get_nc1(mean_vec_list, nc1_feature_dict)
                prob = torch.ones_like(cov_tensor).to(self.device) - nc1_tensor / whole_cov_value
                print("prob")
                print(prob)
                
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                
                features, proj_output = self.model(x)
                features = self.pre_logits(features)
                
                if self.use_residual:
                    # |z-z(i)|**2
                    w_i_lists = -torch.norm(features.view(-1, 1, features.shape[1]) - feature_list, p=2, dim=2)
                    
                    # top_k w_i index select
                    w_i_indexs = torch.topk(w_i_lists, self.knn_top_k)[1].long()
                    
                    # meshgrid로 격자점을 생성
                    idx1, _ = torch.meshgrid(torch.arange(w_i_indexs.shape[0]), torch.arange(w_i_indexs.shape[1]))
                    w_i_lists = self.softmax2(w_i_lists[idx1, w_i_indexs] / self.knn_sigma)

                    # select top_k residuals
                    residual_lists = residual_list[w_i_indexs]
                    residual_terms = torch.bmm(w_i_lists.unsqueeze(1), residual_lists).squeeze()
                    
                if self.use_residual:
                    if self.residual_strategy == "within" or self.residual_strategy == "nc1":
                        index = (prob > torch.rand(1).to(self.device)).nonzero(as_tuple=True)[0]
                        mask = torch.isin(y, index)
                        print("mask", (mask==1).sum())
                        residual_terms *= mask.unsqueeze(1)
                    features += residual_terms
                        
                if self.loss_criterion == "DR":
                    target = self.etf_vec[:, y].t()
                    loss = self.criterion(features, target)

                elif self.loss_criterion == "CE":
                    logit = features @ self.etf_vec
                    loss = criterion(logit, y)

                # accuracy calculation
                with torch.no_grad():
                    cls_score = features @ self.etf_vec
                    pred = torch.argmax(cls_score, dim=-1)
                    _, correct_count = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                    total_correct += correct_count

                    total_loss += loss.item()
                    total_num_data += y.size(0)

                    xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                    correct_l += correct_xlabel_cnt.detach()
                    num_data_l += xlabel_cnt.detach()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).cpu().numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret
