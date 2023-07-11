# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from methods.cl_manager import CLManagerBase
from utils.train_utils import DR_loss, Accuracy, DR_Reverse_loss
import torch
import torch.nn as nn
import copy
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

class ETF_ER_RESMEM_VER7(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.in_channels = self.model.fc.in_features
        # 아래 것들 config 파일에 추가!
        # num class = 100, eval_class = 60 
        self.num_classes = kwargs["num_class"]
        
        if self.ood_strategy == "rotate" and self.use_synthetic_regularization:
            self.num_classes = self.num_classes * 4
            
        self.eval_classes = 0 #kwargs["num_eval_class"]
        self.cls_feature_length = 50
        self.feature_mean_dict = {}
        self.current_cls_feature_dict = {}
        self.feature_std_mean_list = []
        self.current_feature_num = kwargs["current_feature_num"]
        self.residual_num = kwargs["residual_num"]
        self.ood_num_samples = kwargs["ood_num_samples"]
        self.residual_num_threshold = kwargs["residual_num_threshold"]
        self.distill_beta = kwargs["distill_beta"]
        self.distill_strategy = kwargs["distill_strategy"]
        self.distill_threshold = kwargs["distill_threshold"]
        self.use_residual = kwargs["use_residual"]
        self.residual_strategy = kwargs["residual_strategy"]
        self.use_residual_unique = kwargs["use_residual_unique"]
        self.use_residual_warmup = kwargs["use_residual_warmup"]
        self.use_modified_knn = kwargs["use_modified_knn"]
        self.use_patch_permutation = kwargs["use_patch_permutation"]
        self.stds_list = []
        self.masks = {}
        self.residual_dict_index={}
        self.softmax = nn.Softmax(dim=0).to(self.device)
        self.use_feature_distillation = kwargs["use_feature_distillation"]
        
        self.selfsup_temp = kwargs["selfsup_temp"]
        self.selfsup_criterion = SupConLoss(temperature=self.selfsup_temp).to(self.device)
        
        if self.loss_criterion == "DR":
            if self.use_feature_distillation:
                self.criterion = DR_loss(reduction="none").to(self.device)
            else:
                self.criterion = DR_loss().to(self.device)
        elif self.loss_criterion == "CE":
            if self.use_feature_distillation:
                self.criterion = nn.CrossEntropyLoss(reduction="none").to(self.device)
            else:
                self.criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.moco_criterion = nn.CrossEntropyLoss().to(self.device)
        self.regularization_criterion = DR_Reverse_loss(reduction="mean").to(self.device)
        self.compute_accuracy = Accuracy(topk=self.topk)

        # MOCO parameters
        self.moco_k = kwargs["moco_k"]
        self.moco_dim = kwargs["moco_dim"]
        self.moco_T = kwargs["moco_T"]
        self.moco_m = kwargs["moco_m"]
        self.moco_coeff = kwargs["moco_coeff"]

        self.use_neck_forward = kwargs["use_neck_forward"]
        print("self.use_neck_forward", self.use_neck_forward)
        self.model = select_moco_model(self.model_name, self.dataset, 1, pre_trained=False, Neck=self.use_neck_forward, K=self.moco_k, dim=self.moco_dim).to(self.device)
        #self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes).to(self.device)
        
        # moco initialize
        self.ema_model = copy.deepcopy(self.model) #select_moco_model(self.model_name, self.dataset, 1, pre_trained=False, Neck=self.use_neck_forward).to(self.device)
        for param_q, param_k in zip(
            self.model.parameters(), self.ema_model.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        

            
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.etf_initialize()
        self.residual_dict = {}
        self.feature_dict = {}
        self.cls_feature_dict = {}
        self.current_cls_feature_dict_index = {}
        self.note = kwargs["note"]
        self.scl_coeff = kwargs["scl_coeff"]
        os.makedirs(f"{self.note}", exist_ok=True)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.moco_m + param_q.data * (1.0 - self.moco_m)
            

    def get_cos_sim(self, a, b):
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        return cos

    def get_within_class_covariance(self, mean_vec_list, feature_dict):
        cov_tensor = torch.zeros(len(list(mean_vec_list.keys()))).to(self.device)
        # feature dimension 512로 fixed
        for idx, klass in enumerate(list(feature_dict.keys())):
            W = torch.zeros(self.model.fc.in_features, self.model.fc.in_features).to(self.device)
            total_num = 0
            for i in range(len(feature_dict[klass])):
                W += torch.outer((feature_dict[klass][i] - mean_vec_list[klass]), (feature_dict[klass][i] - mean_vec_list[klass]))
            total_num += len(feature_dict[klass])
            W /= total_num
            cov_tensor[idx] = torch.trace(W)
        return cov_tensor

    def get_nc1(self, mean_vec_list, feature_dict):
        nc1_tensor = torch.zeros(len(list(mean_vec_list.keys()))).to(self.device)
        # for global avg calcuation, not just avg mean_vec, feature mean directly (since it is imbalanced dataset)
        total_feature_dict = []
        for key in feature_dict.keys():
            total_feature_dict.extend(feature_dict[key])
        global_mean_vec = torch.mean(torch.stack(total_feature_dict, dim=0), dim=0)


        for idx, klass in enumerate(list(feature_dict.keys())):
            W = torch.zeros(self.model.fc.in_features, self.model.fc.in_features).to(self.device)
            total_num = 0
            for i in range(len(feature_dict[klass])):
                W += torch.outer((feature_dict[klass][i] - mean_vec_list[klass]), (feature_dict[klass][i] - mean_vec_list[klass]))
            total_num += len(feature_dict[klass])
            W /= total_num
            B = torch.outer((mean_vec_list[klass] - global_mean_vec), (mean_vec_list[klass] - global_mean_vec))
            nc1_value = torch.trace(W @ torch.linalg.pinv(B)) / len(mean_vec_list.keys())
            nc1_tensor[idx] = nc1_value
            
        return  nc1_tensor

    def get_within_whole_class_covariance(self, whole_mean_vec, feature_list):
        # feature dimension 512로 fixed
        W = torch.zeros(self.model.fc.in_features, self.model.fc.in_features).to(self.device)
        total_num = 0
        for feature in feature_list:
            W += torch.outer((feature - whole_mean_vec), (feature - whole_mean_vec))
        total_num += len(feature_list)
        W /= total_num
        return torch.trace(W)

    def sample_inference(self, samples):
        with torch.no_grad():
            self.model.eval()
            batch_labels = []
            batch_feature_dict = {}
            for sample in samples:
                x = load_data(sample, self.data_dir, self.test_transform).unsqueeze(0)
                y = self.cls_dict[sample['klass']]
                batch_labels.append(y)
                x = x.to(self.device)
                sample_feature, _ = self.model(x)

                if y not in batch_feature_dict.keys():
                    batch_feature_dict[y] = [sample_feature]
                else:
                    batch_feature_dict[y].append(sample_feature)

            for y in list(set(batch_labels)):
                if y not in self.cls_feature_dict.keys():
                    self.cls_feature_dict[y] = torch.mean(torch.stack(batch_feature_dict[y]), dim=0)
                else:
                    self.cls_feature_dict[y] = self.distill_coeff * self.cls_feature_dict[y] + (1-self.distill_coeff) * torch.mean(torch.stack(batch_feature_dict[y]), dim=0)

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
        self.update_memory(sample, self.future_sample_num)
        self.temp_future_batch.append(sample)
        self.future_num_updates += self.online_iter

        if self.future_num_updates >= 1:
            self.temp_future_batch = []
            self.generate_waiting_batch(int(self.future_num_updates))
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0
    
    '''
    def model_forward(self, x, y, sample_nums, augmented_input=False):
        
        """Forward training data."""

        with torch.cuda.amp.autocast(self.use_amp):
            target = self.etf_vec[:, y].t()
            feature, _ = self.model(x, get_feature=True)
            feature = self.pre_logits(feature)

            if self.loss_criterion == "DR":
                loss = self.criterion(feature, target)
                residual = (target - feature).detach()
                
            elif self.loss_criterion == "CE":
                logit = feature @ self.etf_vec
                loss = self.criterion(logit, y)
                residual = (target - feature).detach()
                #residual = (F.one_hot(y, num_classes=self.num_learned_class) - self.softmax(logit/self.softmax_temperature)).detach()

            if self.use_feature_distillation:
                # calcualte current feature
                for idx, y_i in enumerate(y):
                    if y_i not in self.current_cls_feature_dict.keys():
                        self.current_cls_feature_dict[y_i.item()] = [feature[idx].detach()]
                    else:
                        self.current_cls_feature_dict[y_i.item()].append(feature[idx].detach())
                            
                # check over storing
                for y_i in torch.unique(y):
                    if len(self.current_cls_feature_dict[y_i.item()]) >= self.current_feature_num:
                        self.current_cls_feature_dict[y_i.item()] = self.current_cls_feature_dict[y_i][-self.current_cls_feature_num:]

            if not augmented_input:
                if self.use_feature_distillation:
                    past_feature = torch.stack([self.pre_logits(self.cls_feature_dict[y_i.item()]) for y_i in y], dim=0)
                    target_fc = torch.stack([self.etf_vec[:, y_i.item()] for y_i in y], dim=0)
                    l2_loss = ((feature - past_feature.detach().squeeze()) ** 2).sum(dim=1)
                    # naive distllation
                    if self.distill_strategy == "naive":
                        print("loss", loss.mean(), "l2_loss", self.distill_beta * l2_loss.mean())
                        loss = (loss + self.distill_beta * l2_loss).mean()

                    # similarity based distillation
                    elif self.distill_strategy == "classwise":
                        past_cos_sim = torch.abs(self.get_cos_sim(target_fc, past_feature.detach().squeeze(dim=1)))
                        past_cos_sim -= self.distill_threshold
                        beta_masking = torch.clamp(past_cos_sim, min=0.0, max=1.0)
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()
                    
                    elif self.distill_strategy == "classwise_difference":
                        current_feature = torch.stack([self.pre_logits(torch.mean(torch.stack(self.current_cls_feature_dict[y_i.item()], dim=0), dim=0).unsqueeze(0)) for y_i in y], dim=0)
                        past_cos_sim = self.get_cos_sim(target_fc, past_feature.detach().squeeze(dim=1))
                        current_cos_sim = self.get_cos_sim(target_fc, current_feature.squeeze(dim=1).to(self.device))
                        masking = torch.abs(past_cos_sim) > torch.abs(current_cos_sim)
                        beta_masking = torch.abs(past_cos_sim)
                        beta_masking -= self.distill_threshold
                        beta_masking = torch.clamp(beta_masking, min=0.0, max=1.0)
                        beta_masking = masking * beta_masking
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()
                
                    elif self.distill_strategy == "classwise_difference_ver2":
                        current_feature = torch.stack([self.pre_logits(torch.mean(torch.stack(self.current_cls_feature_dict[y_i.item()], dim=0), dim=0).unsqueeze(0)) for y_i in y], dim=0)
                        past_cos_sim = self.get_cos_sim(target_fc, past_feature.detach().squeeze(dim=1))
                        current_cos_sim = self.get_cos_sim(target_fc, current_feature.squeeze(dim=1).to(self.device))
                        masking = torch.abs(past_cos_sim) > torch.abs(current_cos_sim)
                        beta_masking = torch.abs(past_cos_sim) - torch.abs(current_cos_sim)
                        beta_masking -= self.distill_threshold
                        beta_masking = torch.clamp(beta_masking, min=0.0, max=1.0)
                        beta_masking = masking * beta_masking
                        #sim_difference = torch.abs(past_cos_sim - current_cos_sim)
                        #sim_difference -= self.distill_threshold
                        #beta_masking = torch.clamp(sim_difference, min=0.0, max=1.0)
                        print("y")
                        print(y)
                        print("beta_masking")
                        print(beta_masking)
                        print("loss", loss.mean(), "l2_loss", (self.distill_beta * beta_masking * l2_loss).mean())
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()

                # residual dict update
                if self.use_residual:
                    if self.use_residual_unique:    
                        for idx, t in enumerate(y):
                            if t.item() not in self.residual_dict.keys():
                                self.residual_dict[t.item()] = [residual[idx]]
                                self.feature_dict[t.item()] = [feature.detach()[idx]]
                                self.residual_dict_index[t.item()] = [sample_nums[idx].item()]
                            else: 
                                if sample_nums[idx].item() in self.residual_dict_index[t.item()]:
                                    target_index = self.residual_dict_index[t.item()].index(sample_nums[idx].item())
                                    del self.residual_dict[t.item()][target_index]
                                    del self.feature_dict[t.item()][target_index]
                                    del self.residual_dict_index[t.item()][target_index]
                                    
                                self.residual_dict[t.item()].append(residual[idx])
                                self.feature_dict[t.item()].append(feature.detach()[idx])
                                self.residual_dict_index[t.item()].append(sample_nums[idx].item())
                                
                            if len(self.residual_dict[t.item()]) > self.residual_num:
                                self.residual_dict[t.item()] = self.residual_dict[t.item()][1:]
                                self.feature_dict[t.item()] = self.feature_dict[t.item()][1:]
                                self.residual_dict_index[t.item()] = self.residual_dict_index[t.item()][1:]
                    else:
                        for idx, t in enumerate(y):
                            if t.item() not in self.residual_dict.keys():
                                self.residual_dict[t.item()] = [residual[idx]]
                                self.feature_dict[t.item()] = [feature.detach()[idx]]
                            else:  
                                self.residual_dict[t.item()].append(residual[idx])
                                self.feature_dict[t.item()].append(feature.detach()[idx])
                                
                            if len(self.residual_dict[t.item()]) > self.residual_num:
                                self.residual_dict[t.item()] = self.residual_dict[t.item()][1:]
                                self.feature_dict[t.item()] = self.feature_dict[t.item()][1:]

            # accuracy calculation
            with torch.no_grad():
                cls_score = feature.detach() @ self.etf_vec
                if self.use_synthetic_regularization:
                    acc, correct = self.compute_accuracy(cls_score, y, real_entered_num_class = len(self.memory.cls_list), real_num_class = self.real_num_classes)
                else:    
                    acc, correct = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                
                acc = acc.item()
        
        return loss, feature, correct
    '''
    
    def model_forward(self, x, y, sample_nums, augmented_input=False):
        
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
                #residual = (F.one_hot(y, num_classes=self.num_learned_class) - self.softmax(logit/self.softmax_temperature)).detach()
            
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
            # print("loss", loss, "selfsup_loss", selfsup_loss)
            loss += (self.moco_coeff * selfsup_loss)
    
                         
            if self.use_feature_distillation:
                # calcualte current feature
                for idx, y_i in enumerate(y):
                    if y_i not in self.current_cls_feature_dict.keys():
                        self.current_cls_feature_dict[y_i.item()] = [feature[idx].detach()]
                    else:
                        self.current_cls_feature_dict[y_i.item()].append(feature[idx].detach())
                            
                # check over storing
                for y_i in torch.unique(y):
                    if len(self.current_cls_feature_dict[y_i.item()]) >= self.current_feature_num:
                        self.current_cls_feature_dict[y_i.item()] = self.current_cls_feature_dict[y_i][-self.current_cls_feature_num:]

            if not augmented_input:
                if self.use_feature_distillation:
                    past_feature = torch.stack([self.pre_logits(self.cls_feature_dict[y_i.item()]) for y_i in y], dim=0)
                    target_fc = torch.stack([self.etf_vec[:, y_i.item()] for y_i in y], dim=0)
                    l2_loss = ((feature - past_feature.detach().squeeze()) ** 2).sum(dim=1)
                    # naive distllation
                    if self.distill_strategy == "naive":
                        print("loss", loss.mean(), "l2_loss", self.distill_beta * l2_loss.mean())
                        loss = (loss + self.distill_beta * l2_loss).mean()

                    # similarity based distillation
                    elif self.distill_strategy == "classwise":
                        past_cos_sim = torch.abs(self.get_cos_sim(target_fc, past_feature.detach().squeeze(dim=1)))
                        past_cos_sim -= self.distill_threshold
                        beta_masking = torch.clamp(past_cos_sim, min=0.0, max=1.0)
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()
                    
                    elif self.distill_strategy == "classwise_difference":
                        current_feature = torch.stack([self.pre_logits(torch.mean(torch.stack(self.current_cls_feature_dict[y_i.item()], dim=0), dim=0).unsqueeze(0)) for y_i in y], dim=0)
                        past_cos_sim = self.get_cos_sim(target_fc, past_feature.detach().squeeze(dim=1))
                        current_cos_sim = self.get_cos_sim(target_fc, current_feature.squeeze(dim=1).to(self.device))
                        masking = torch.abs(past_cos_sim) > torch.abs(current_cos_sim)
                        beta_masking = torch.abs(past_cos_sim)
                        beta_masking -= self.distill_threshold
                        beta_masking = torch.clamp(beta_masking, min=0.0, max=1.0)
                        beta_masking = masking * beta_masking
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()
                
                    elif self.distill_strategy == "classwise_difference_ver2":
                        current_feature = torch.stack([self.pre_logits(torch.mean(torch.stack(self.current_cls_feature_dict[y_i.item()], dim=0), dim=0).unsqueeze(0)) for y_i in y], dim=0)
                        past_cos_sim = self.get_cos_sim(target_fc, past_feature.detach().squeeze(dim=1))
                        current_cos_sim = self.get_cos_sim(target_fc, current_feature.squeeze(dim=1).to(self.device))
                        masking = torch.abs(past_cos_sim) > torch.abs(current_cos_sim)
                        beta_masking = torch.abs(past_cos_sim) - torch.abs(current_cos_sim)
                        beta_masking -= self.distill_threshold
                        beta_masking = torch.clamp(beta_masking, min=0.0, max=1.0)
                        beta_masking = masking * beta_masking
                        #sim_difference = torch.abs(past_cos_sim - current_cos_sim)
                        #sim_difference -= self.distill_threshold
                        #beta_masking = torch.clamp(sim_difference, min=0.0, max=1.0)
                        print("y")
                        print(y)
                        print("beta_masking")
                        print(beta_masking)
                        print("loss", loss.mean(), "l2_loss", (self.distill_beta * beta_masking * l2_loss).mean())
                        loss = (loss + self.distill_beta * beta_masking * l2_loss).mean()

                # residual dict update
                if self.use_residual:
                    if self.use_residual_unique:    
                        for idx, t in enumerate(y):
                            if t.item() not in self.residual_dict.keys():
                                self.residual_dict[t.item()] = [residual[idx]]
                                self.feature_dict[t.item()] = [feature.detach()[idx]]
                                self.residual_dict_index[t.item()] = [sample_nums[idx].item()]
                            else: 
                                if sample_nums[idx].item() in self.residual_dict_index[t.item()]:
                                    target_index = self.residual_dict_index[t.item()].index(sample_nums[idx].item())
                                    del self.residual_dict[t.item()][target_index]
                                    del self.feature_dict[t.item()][target_index]
                                    del self.residual_dict_index[t.item()][target_index]
                                    
                                self.residual_dict[t.item()].append(residual[idx])
                                self.feature_dict[t.item()].append(feature.detach()[idx])
                                self.residual_dict_index[t.item()].append(sample_nums[idx].item())
                                
                            if len(self.residual_dict[t.item()]) > self.residual_num:
                                self.residual_dict[t.item()] = self.residual_dict[t.item()][1:]
                                self.feature_dict[t.item()] = self.feature_dict[t.item()][1:]
                                self.residual_dict_index[t.item()] = self.residual_dict_index[t.item()][1:]
                    else:
                        for idx, t in enumerate(y):
                            if t.item() not in self.residual_dict.keys():
                                self.residual_dict[t.item()] = [residual[idx]]
                                self.feature_dict[t.item()] = [feature.detach()[idx]]
                            else:  
                                self.residual_dict[t.item()].append(residual[idx])
                                self.feature_dict[t.item()].append(feature.detach()[idx])
                                
                            if len(self.residual_dict[t.item()]) > self.residual_num:
                                self.residual_dict[t.item()] = self.residual_dict[t.item()][1:]
                                self.feature_dict[t.item()] = self.feature_dict[t.item()][1:]
                
            # accuracy calculation
            with torch.no_grad():
                cls_score = feature.detach() @ self.etf_vec
                if self.use_synthetic_regularization:
                    acc, correct = self.compute_accuracy(cls_score, y, real_entered_num_class = len(self.memory.cls_list), real_num_class = self.real_num_classes)
                else:    
                    acc, correct = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                
                acc = acc.item()
        
        return loss, feature, correct
    
    
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

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

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            sample_nums = data["sample_nums"].to(self.device)
            self.before_model_update()
            self.optimizer.zero_grad()

            # logit can not be used anymore
            loss, feature, correct_batch = self.model_forward(x,y, sample_nums)
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.after_model_update()

            total_loss += loss.item()
            correct += correct_batch
            num_data += y.size(0)
            
            if self.use_synthetic_regularization:
                ood_data = self.ood_inference(self.ood_num_samples)
                if ood_data is not None:
                    new_x, new_y = ood_data
                    self.optimizer.zero_grad()
                    loss, feature, correct_batch = self.model_forward(new_x, new_y, sample_nums, augmented_input=True)
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                #self.after_model_update()
                #total_loss += loss.item()            

        return total_loss / iterations, correct / num_data
        
    def etf_initialize(self):
        logger.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))
        orth_vec = self.generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        self.etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1))).to(self.device)
        print("self.etf_vec", self.etf_vec.shape)

            
    def get_angle(self, a, b):
        inner_product = (a * b).sum(dim=0)
        a_norm = a.pow(2).sum(dim=0).pow(0.5)
        b_norm = b.pow(2).sum(dim=0).pow(0.5)
        cos = inner_product / (2 * a_norm * b_norm)
        angle = torch.acos(cos)
        return angle

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        rand_mat = np.random.random(size=(feat_in, num_classes))
        orth_vec, _ = np.linalg.qr(rand_mat) # qr 분해를 통해서 orthogonal한 basis를 get
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
            "The max irregular value is : {}".format(
                torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
        return orth_vec


    def get_mean_vec(self):
        feature_dict = {}
        mean_vec_list = {}
        mean_vec_tensor_list = []
        mean_vec_tensor_list = torch.zeros(len(list(self.feature_dict.keys())), self.model.fc.in_features).to(self.device)
        for cls in list(self.feature_dict.keys()):
            feature_dict[cls] = torch.stack(self.feature_dict[cls]).detach()
            feature_dict[cls] /= torch.norm(torch.stack(self.feature_dict[cls], dim=0), p=2, dim=1, keepdim=True)
            mean_vec_list[cls] = torch.mean(torch.stack(self.feature_dict[cls]), dim=0)
            mean_vec_tensor_list[cls] = mean_vec_list[cls]
        
        return mean_vec_tensor_list

    def ood_store(self, ood_dict):
        name_prefix = self.note + "/etf_resmem_sigma" + str(self.sigma) + "_num_" + str(self.sample_num) + "_iter" + str(self.online_iter) + "_sigma" + str(self.softmax_temperature) + "_criterion_" + self.select_criterion + "_top_k" + str(self.knn_top_k) + "_knn_sigma"+ str(self.knn_sigma)
        ood_pickle_name = name_prefix + "_ood.pickle"        

        for key in list(ood_dict.keys()):
            ood_dict[key] = torch.cat(ood_dict[key])
            
        with open(ood_pickle_name, 'wb') as f:
            pickle.dump(ood_dict, f, pickle.HIGHEST_PROTOCOL) 

    def ood_inference(self, ood_num_samples, ood_sampling_num=1):
        ood_dict = {}
        new_x_data = []
        new_y_data = []
        
        feature_cluster = self.get_mean_vec()
        loss = None
        #for ood_data in ood_data_list:
        #ood_data = self.memory.generate_ood_class(num_samples=int(self.batch_size/4))
        ood_data_list = [self.memory.generate_ood_class(num_samples=ood_num_samples) for _ in range(ood_sampling_num)]
        for ood_data in ood_data_list:
            if ood_data is not None:
                if self.ood_strategy == "cutmix":
                    key, x1, x2 = ood_data
                    x1 = load_batch(x1, self.data_dir, self.test_transform)
                    x2 = load_batch(x2, self.data_dir, self.test_transform)
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    
                    if key not in self.masks.keys():
                        self.masks[key] = generate_masking(x1, self.device)
                    #if key not in ood_dict.keys():
                    #    ood_dict[key] = []
                    index, mask = self.masks[key]

                    if self.use_patch_permutation:
                        new_x1, _ = generate_new_data(x1, x2, self.device, mask, index)
                    else:
                        new_x1, _ = generate_new_data(x1, x2, self.device, mask)

                    new_x_data.append(new_x1)
                    #new_y_data.append(new_y1) # TODO new_y1을 cutmix에서 어떻게 define할지

                    '''
                    _, feature = self.model(new_x1, get_feature=True)
                    feature = self.pre_logits(feature)            
                    loss = self.regularization_criterion(feature, feature_cluster, self.num_learned_class)
                    ood_dict[key].append(feature)
                    '''
                
                elif self.ood_strategy == "rotate":
                    x, y = ood_data
                    x = load_batch(x, self.data_dir, self.test_transform)
                    x = x.to(self.device)
                    
                    new_x, new_y = my_segmentation_transforms(x, y, self.real_num_classes)
                    new_y = new_y.to(self.device)
                    new_x_data.append(new_x)
                    new_y_data.append(new_y)
                    '''
                    _, feature_right = self.model(right_x, get_feature=True)
                    feature_right = self.pre_logits(feature_right)
                    
                    _, feature_left = self.model(left_x, get_feature=True)
                    feature_left = self.pre_logits(feature_left)
                    
                    right_key = str(cls) + "_right"
                    left_key = str(cls) + "_left"
                    if right_key not in ood_dict.keys():
                        ood_dict[right_key] = []
                        ood_dict[left_key] = []
                    ood_dict[right_key].append(feature_right)
                    ood_dict[left_key].append(feature_left)
                    '''
                    
        #return ood_dict, loss
        if len(new_x_data) == 0:
            return None
        else:
            return torch.cat(new_x_data), torch.cat(new_y_data)
    
    def update_memory(self, sample, sample_num=None):
        #self.reservoir_memory(sample)
        self.balanced_replace_memory(sample, sample_num)

    def add_new_class(self, class_name, sample):
        self.added = True
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample)
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            if self.use_feature_distillation:
                self.sample_inference([sample])
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

        # save feature and etf-fc
        if self.store_pickle and self.rnd_seed == 1:
            if self.sample_num % 100 == 0 and self.sample_num !=0:

                name_prefix = self.note + "/etf_resmem_sigma" + str(self.sigma) + "_num_" + str(self.sample_num) + "_iter" + str(self.online_iter) + "_sigma" + str(self.softmax_temperature) + "_criterion_" + self.select_criterion + "_top_k" + str(self.knn_top_k) + "_knn_sigma"+ str(self.knn_sigma)
                fc_pickle_name = name_prefix + "_fc.pickle"
                feature_pickle_name = name_prefix + "_feature.pickle"
                class_pickle_name = name_prefix + "_class.pickle"
                pickle_name_feature_std_mean_list = name_prefix + "_feature_std.pickle"
                pickle_name_stds_list = name_prefix + "_stds.pickle"

                self.save_features(feature_pickle_name, class_pickle_name)

                with open(fc_pickle_name, 'wb') as f:
                    '''
                    num_leanred_class = len(self.memory.cls_list)
                    index = []
                    for i in range(4):
                        #inf_index += list(range(i * real_num_class, i * real_num_class + real_entered_num_class))
                        index += list(range(i * self.real_num_classes + num_leanred_class, min((i+1) * self.real_num_classes, self.num_classes)))
                    pickle.dump(self.etf_vec[:, index].T, f, pickle.HIGHEST_PROTOCOL)
                    '''
                    pickle.dump(self.etf_vec[:, :len(self.memory.cls_list)].T, f, pickle.HIGHEST_PROTOCOL)

                with open(pickle_name_feature_std_mean_list, 'wb') as f:
                    pickle.dump(self.feature_std_mean_list, f, pickle.HIGHEST_PROTOCOL)
                
                with open(pickle_name_stds_list, 'wb') as f:
                    pickle.dump(self.stds_list, f, pickle.HIGHEST_PROTOCOL)
                
                # ood data 저장
                '''
                if self.ood_strategy != "none": 
                    self.ood_inference()
                '''
        

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)


    def sub_simple_test(self, x, softmax=False, post_process=False):
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        #cls_score = cls_score[:, :self.eval_classes]
        cls_score = cls_score[:, :len(self.memory.cls_list)]
        assert not softmax
        '''
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score
        '''
        return cls_score


    def simple_test(self, img, gt_label, return_feature=True):
        """Test without augmentation."""
        '''
        if return_backbone:
            x = self.extract_feat(img, stage='backbone')
            return x
        x = self.extract_feat(img)
        '''
        feature, _ = self.model(img)
        res = self.sub_simple_test(feature, post_process=False)
        res = res.argmax(dim=-1)
        '''
        if return_feature:
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist(), feature
        else:
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist()
        '''
        if return_feature:
            return torch.eq(res, gt_label).to(dtype=torch.float32), feature
        else:
            return torch.eq(res, gt_label).to(dtype=torch.float32)

    def evaluation(self, test_loader, criterion):
        print("Memory State")
        print(self.memory.cls_count)
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes).to(self.device)
        num_data_l = torch.zeros(self.n_classes).to(self.device)
        total_acc = 0.0
        label = []
        feature_dict = {}
        self.model.eval()

        if self.use_residual:
            residual_list = []
            feature_list = []
            key_list = []
            for key in self.residual_dict.keys():
                residual_list.extend(self.residual_dict[key])
                feature_list.extend(self.feature_dict[key])
                key_list.extend([int(key) for _ in range(len(self.feature_dict[key]))])
                print("residual", key, len(self.residual_dict[key]))
            residual_list = torch.stack(residual_list)
            feature_list = torch.stack(feature_list)
            key_list = torch.Tensor(key_list).to(self.device)

            # residual dict 내의 feature들이 어느정도 잘 모여있는 상태여야 residual term good
            nc1_feature_dict = {}
            mu_G = 0
            num_feature = 0
            mean_vec_list = {}
            mean_vec_tensor_list = []
            for cls in list(self.feature_dict.keys()):
                nc1_feature_dict[cls] = torch.stack(self.feature_dict[cls]).detach()
                nc1_feature_dict[cls] /= torch.norm(torch.stack(self.feature_dict[cls], dim=0), p=2, dim=1, keepdim=True)
                mean_vec_list[cls] = torch.mean(torch.stack(self.feature_dict[cls]), dim=0)
                mean_vec_tensor_list.append(mean_vec_list[cls])
                mu_G += torch.sum(nc1_feature_dict[cls], dim=0)
                num_feature += len(self.feature_dict[cls])
            
            mu_G /= num_feature
            mean_vec_tensor_list = torch.stack(mean_vec_tensor_list)       
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
                    # print("-torch.norm(feature - feature_list, p=2, dim=1, keepdim=True)", torch.norm(features[0] - feature_list, p=2, dim=1, keepdim=True).shape)
                    w_i_lists = [-torch.norm(feature - feature_list, p=2, dim=1, keepdim=True) for feature in features.detach()]

                    # top_k w_i index select
                    w_i_indexs = [torch.topk(w_i_list.squeeze(), self.knn_top_k)[1].long() for w_i_list in w_i_lists]
                    
                    # modified_knn
                    if self.use_modified_knn:
                        w_i_new_indexs = []
                        for w_i in w_i_indexs:
                            unique = torch.unique(key_list[w_i], return_counts=True)
                            index = (key_list[w_i] == unique[0][torch.argmax(unique[1]).item()].item()).nonzero(as_tuple=True)[0]
                            w_i_new_indexs.append(w_i[index])
                        w_i_indexs = w_i_new_indexs
                        
                    # top_k w_i 
                    if self.select_criterion == "softmax":
                        #w_i_lists = [self.softmax(torch.topk(w_i_list.squeeze(), self.knn_top_k)[0] / self.knn_sigma) for w_i_list in w_i_lists]
                        w_i_lists = [self.softmax(w_i_list.squeeze()[w_i_index] / self.knn_sigma) for w_i_index, w_i_list in zip(w_i_indexs, w_i_lists)]
                    elif self.select_criterion == "linear": # just weight sum
                        w_i_lists = [torch.topk(w_i_list.squeeze(), self.knn_top_k)[0] / torch.sum(torch.topk(w_i_list.squeeze(), self.knn_top_k)[0]).item() for w_i_list in w_i_lists]

                    # select top_k residuals
                    residual_lists = [residual_list[w_i_index] for w_i_index in w_i_indexs]
                    residual_terms = [rs_list.T @ w_i_list  for rs_list, w_i_list in zip(residual_lists, w_i_lists)]

                    '''
                    unique_y = torch.unique(y).tolist()
                    for u_y in unique_y:
                        indices = (y == u_y).nonzero(as_tuple=True)[0]
                        if u_y not in feature_dict.keys():
                            feature_dict[u_y] = [torch.index_select(features, 0, indices)]
                        else:
                            feature_dict[u_y].append(torch.index_select(features, 0, indices))
                    '''


                if self.use_residual:
                    if self.use_residual_unique:
                        total_mask = torch.Tensor([len(self.residual_dict_index[key]) for key in list(self.residual_dict_index.keys())]).to(self.device) >= self.residual_num_threshold
                    else:
                        total_mask = torch.Tensor([len(self.residual_dict[key]) for key in list(self.residual_dict.keys())]).to(self.device) >= self.residual_num_threshold
                        
                    if self.residual_strategy == "within" or self.residual_strategy == "nc1":
                        prob_mask = prob > torch.rand(1).to(self.device)
                        if self.use_residual_warmup:
                            prob_mask *= total_mask
                        mask = torch.zeros(len(y)).to(self.device)
                        for idx, y_i in enumerate(y):
                            if y_i >= len(prob_mask):
                                mask[idx] = 0
                                continue
                            mask[idx] = prob_mask[y_i.item()]
                        index = (mask==1).nonzero(as_tuple=True)[0]
                        features[index] += torch.stack(residual_terms).to(self.device)[index]
                        
                    elif self.residual_strategy == "none":
                        if self.use_residual_warmup:
                            mask = torch.zeros(len(y)).to(self.device)
                            for idx, y_i in enumerate(y):
                                if y_i >= len(total_mask):
                                    mask[idx] = 0
                                    continue
                                mask[idx] = total_mask[y_i.item()]
                            index = (mask==1).nonzero(as_tuple=True)[0]                      
                            features[index] += torch.stack(residual_terms).to(self.device)[index]
                        else:
                            features += torch.stack(residual_terms).to(self.device)
                        
                if self.loss_criterion == "DR":
                    target = self.etf_vec[:, y].t()
                    loss = self.criterion(features, target)

                elif self.loss_criterion == "CE":
                    logit = features @ self.etf_vec
                    loss = criterion(logit, y)

                if self.use_feature_distillation:
                    loss = loss.mean()

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
