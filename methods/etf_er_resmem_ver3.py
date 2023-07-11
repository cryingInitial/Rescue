# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from collections import defaultdict
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
from utils.train_utils import select_optimizer, select_model, select_scheduler
from utils.augment import my_segmentation_transforms
logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class ETF_ER_RESMEM_VER3(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.in_channels = self.model.fc.in_features
        self.num_classes = kwargs["num_class"]
        self.eval_classes = 0 #kwargs["num_eval_class"]
        self.cls_feature_length = 50
        self.feature_mean_dict = {}
        self.current_cls_feature_dict = {}
        self.feature_std_mean_list = []
        self.current_feature_num = kwargs["current_feature_num"]
        self.residual_num = kwargs["residual_num"]
        self.residual_strategy = kwargs["residual_strategy"]
        self.future_train_loader = None
        self.future_test_loader = None
        self.stds_list = []
        self.masks = {}
        self.residual_dict_index={}
        self.softmax2 = nn.Softmax(dim=1).to(self.device)
        self.softmax = nn.Softmax(dim=0).to(self.device)
        
        if self.loss_criterion == "DR":
            self.criterion = DR_loss().to(self.device)
        elif self.loss_criterion == "CE":
            self.criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        self.regularization_criterion = DR_Reverse_loss(reduction="mean").to(self.device)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.use_neck_forward = kwargs["use_neck_forward"]
        print("self.use_neck_forward", self.use_neck_forward)
        self.model = select_model(self.model_name, self.dataset, 1, pre_trained=False, Neck=self.use_neck_forward).to(self.device)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.etf_initialize()
        self.residual_dict = defaultdict(list)
        self.feature_dict = defaultdict(list)
        self.cls_feature_dict = {}
        self.current_cls_feature_dict_index = {}
        self.note = kwargs["note"]
        os.makedirs(f"{self.note}", exist_ok=True)

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def get_cos_sim(self, a, b):
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        return cos

    def batch_cov(self, points):
        B, D = points.size()
        mean = points.mean(dim=0)
        diffs = (points - mean)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1))
        return torch.mean(prods.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1))
    
    def get_within_class_covariance(self, feature_dict):
        cov_tensor = torch.zeros(len(list(feature_dict.keys()))).to(self.device)
        # feature dimension 512로 fixed
        for idx, klass in enumerate(list(feature_dict.keys())):
            cov_tensor[klass] = self.batch_cov(feature_dict[klass])
        return cov_tensor
    
    
    def get_within_class_covariance2(self, mean_vec_list, feature_dict):
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

    def get_within_whole_class_covariance2(self, whole_mean_vec, feature_list):
        # feature dimension 512로 fixed
        W = torch.zeros(self.model.fc.in_features, self.model.fc.in_features).to(self.device)
        total_num = 0
        for feature in feature_list:
            W += torch.outer((feature - whole_mean_vec), (feature - whole_mean_vec))
        total_num += len(feature_list)
        W /= total_num
        return torch.trace(W)

    def get_within_whole_class_covariance(self, whole_features):
        #return torch.stack(sum([v for v in feature_dict.values()], []))
        return self.batch_cov(whole_features)

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
                _, sample_feature = self.model(x, get_feature=True)

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

    def model_forward(self, x, y, sample_nums):
        
        with torch.cuda.amp.autocast(self.use_amp):
            """Forward training data."""
            target = self.etf_vec[:, y].t()

            _, feature = self.model(x, get_feature=True)
            feature = self.pre_logits(feature)

            if self.loss_criterion == "DR":
                loss = self.criterion(feature, target)
                residual = (target - feature).detach()
                
            elif self.loss_criterion == "CE":
                logit = feature @ self.etf_vec
                loss = self.criterion(logit, y)
                residual = (target - feature).detach()

            # residual dict update
            if self.use_residual:
                for label in torch.unique(y):
                    index = (y==label).nonzero(as_tuple=True)[0]
                    self.residual_dict[label.item()].extend(residual[index])
                    self.feature_dict[label.item()].extend(feature.detach()[index])
                    
                    if len(self.residual_dict[label.item()]) > self.residual_num:
                        self.residual_dict[label.item()] = self.residual_dict[label.item()][-self.residual_num:]
                        self.feature_dict[label.item()] = self.feature_dict[label.item()][-self.residual_num:]
            

            # accuracy calculation
            with torch.no_grad():
                cls_score = feature.detach() @ self.etf_vec
                acc, correct = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                
                acc = acc.item()
        
            return loss, feature, correct


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
        _, feature = self.model(img, get_feature=True)
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

    def future_evaluation(self):
        
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes).to(self.device)
        num_data_l = torch.zeros(self.n_classes).to(self.device)
        
        # k-shot training temp_model using future data
        temp_model = copy.deepcopy(self.model)
        temp_model.train()
        temp_optimizer = select_optimizer(self.opt_name, self.lr, temp_model)
        
        for name, param in temp_model.named_parameters():
            if 'neck' not in name:
                param.requires_grad = False
                
        future_residual_dict = defaultdict(list)
        future_feature_dict = defaultdict(list)
        
        for i in range(self.future_training_iterations):
            for i, data in enumerate(self.future_train_loader):
                x = data["image"].to(self.device)
                y = data["label"].to(self.device)
                
                temp_optimizer.zero_grad()
                # logit can not be used anymore
                with torch.cuda.amp.autocast(self.use_amp):
                    target = self.etf_vec[:, y].t()
                    _, feature = temp_model(x, get_feature=True)
                    feature = self.pre_logits(feature)
                    loss = self.criterion(feature, target)
                    residual = (target - feature).detach()
                        
                    if self.use_residual:
                        for idx, t in enumerate(y):
                            future_residual_dict[t.item()].append(residual[idx])
                            future_feature_dict[t.item()].append(feature.detach()[idx])
                                
                            if len(future_residual_dict[t.item()]) > self.residual_num:
                                future_residual_dict[t.item()] = future_residual_dict[t.item()][1:]
                                future_feature_dict[t.item()] = future_feature_dict[t.item()][1:]
                            
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(temp_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    temp_optimizer.step()

        '''
        # merge current (feature, residual) dict and future (feature, residual) dict
        for key in self.residual_dict.keys():
            future_residual_dict[key] = self.residual_dict[key]
            future_feature_dict[key] = self.feature_dict[key]
        '''

        if self.use_residual:
            residual_list = torch.stack(list(future_residual_dict.values())[0])
            feature_list = torch.stack(list(future_feature_dict.values())[0])

            # residual dict 내의 feature들이 어느정도 잘 모여있는 상태여야 residual term good
            nc1_feature_dict = defaultdict(list)
            mean_vec_list = defaultdict(list)
            for cls in list(future_feature_dict.keys()):
                stacked_feature_dict = torch.stack(future_feature_dict[cls]).detach()
                nc1_feature_dict[cls] = stacked_feature_dict / torch.norm(stacked_feature_dict, p=2, dim=1, keepdim=True)
                mean_vec_list[cls] = torch.mean(stacked_feature_dict, dim=0)
                
            mu_G = torch.mean(torch.stack(list(future_feature_dict.values())[0]), dim=0)
            whole_cov_value2 = self.get_within_whole_class_covariance2(mu_G, feature_list)
            whole_cov_value = self.get_within_whole_class_covariance(mu_G, feature_list)
            print("whole_cov_value2", whole_cov_value2, "whole_cov_value", whole_cov_value)
            if self.residual_strategy == "within":
                cov_tensor = self.get_within_class_covariance(mean_vec_list, nc1_feature_dict)
                prob = torch.ones_like(cov_tensor).to(self.device) - cov_tensor / whole_cov_value
                print("future prob")
                print(prob)
            elif self.residual_strategy == "nc1":
                nc1_tensor = self.get_nc1(mean_vec_list, nc1_feature_dict)
                prob = torch.ones_like(cov_tensor).to(self.device) - nc1_tensor / whole_cov_value
                print("future prob")
                print(prob)
        
        temp_model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.future_test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                _, features = temp_model(x, get_feature=True)
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
                        residual_terms *= mask.unsqueeze(1)
                        
                    features += residual_terms
                        
                if self.loss_criterion == "DR":
                    target = self.etf_vec[:, y].t()
                    loss = self.criterion(features, target)

                elif self.loss_criterion == "CE":
                    logit = features @ self.etf_vec
                    loss = self.criterion(logit, y)


                cls_score = features @ self.etf_vec
                pred = torch.argmax(cls_score, dim=-1)
                #_, correct_count = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list)], y)
                
                _, correct_count = self.compute_accuracy(cls_score[:, :len(self.memory.cls_list) + self.num_future_class], y)
                total_correct += correct_count

                total_loss += loss.item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach()
                num_data_l += xlabel_cnt.detach()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(self.test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).cpu().numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return ret
        
        
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
            norm_feature_value = torch.norm(feature_list, p=2, dim=0, keepdim=True)
            
            # residual dict 내의 feature들이 어느정도 잘 모여있는 상태여야 residual term good
            nc1_feature_dict = defaultdict(list)
            for cls in list(self.feature_dict.keys()):
                nc1_feature_dict[cls] = torch.stack(self.feature_dict[cls]) / norm_feature_value
            
            if self.residual_strategy == "within":
                whole_cov_value = self.get_within_whole_class_covariance(feature_list / norm_feature_value)
                cov_tensor = self.get_within_class_covariance(nc1_feature_dict)
                prob = torch.ones_like(cov_tensor).to(self.device) - cov_tensor / whole_cov_value
            '''
            elif self.residual_strategy == "nc1":
                nc1_tensor = self.get_nc1(mean_vec_list, nc1_feature_dict)
                prob = torch.ones_like(cov_tensor).to(self.device) - nc1_tensor / whole_cov_value
                print("prob")
                print(prob)
            '''
                
        with torch.no_grad():
            rand_num = torch.rand(1).to(self.device)
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                _, features = self.model(x, get_feature=True)
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
                        index = (prob > rand_num).nonzero(as_tuple=True)[0]
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
