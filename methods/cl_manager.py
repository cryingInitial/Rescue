import logging
import os
import copy
import pickle
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from utils.augment import my_segmentation_transforms
from flops_counter.ptflops import get_model_complexity_info
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics
from utils.augment import get_transform
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLManagerBase:
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
       
        self.device = device
        self.added = True
        self.writer = SummaryWriter(f'tensorboard/{kwargs["dataset"]}/{kwargs["note"]}/seed_{kwargs["rnd_seed"]}')
        self.method_name = kwargs["mode"]
        self.dataset = kwargs["dataset"]
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.init_cls = kwargs["init_cls"]
        self.use_residual = kwargs["use_residual"]
        self.memory_size = kwargs["memory_size"]
        self.online_iter = kwargs["online_iter"]
        self.std_dict = {}
        self.btw_std_dict = []
        self.dist_dict = {}
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'const'
        self.lr = kwargs["lr"]
        self.real_num_classes = kwargs["num_class"]
        # assert kwargs["temp_batchsize"] <= kwargs["batchsize"]
        # self.batch_size = kwargs["batchsize"]
        # self.temp_batch_size = kwargs["temp_batchsize"]
        # self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.batch_size = 21
        self.temp_batch_size = 8
        self.memory_batch_size = 20
        self.memory_size -= self.temp_batch_size
        self.transforms = kwargs["transforms"]
        self.num_future_class = kwargs["num_future_class"]
        self.criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.data_dir = kwargs["data_dir"]
        if self.data_dir is None:
            self.data_dir = os.path.join("dataset", self.dataset)
        self.n_worker = kwargs["n_worker"]
        self.future_steps = kwargs["future_steps"]
        self.transform_on_gpu = kwargs["transform_on_gpu"]
        self.use_kornia = kwargs["use_kornia"]
        self.transform_on_worker = kwargs["transform_on_worker"]
        self.eval_period = kwargs["eval_period"]
        self.topk = kwargs["topk"]
        self.f_period = kwargs["f_period"]
        self.use_amp = kwargs["use_amp"]
        self.future_training_iterations = kwargs["future_training_iterations"]
        self.use_future_eval = kwargs["use_future_eval"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.cls_dict = {}
        self.total_samples = len(self.train_datalist)

        self.train_transform, self.test_transform, self.cpu_transform, self.test_gpu_transform, self.future_train_transform, self.n_classes = get_transform(self.dataset, self.transforms, self.transform_on_gpu)
        self.cutmix = "cutmix" in kwargs["transforms"]

        self.model = select_model(self.model_name, self.dataset, pre_trained=True ).to(self.device)
        print("model")
        print(self.model)
        self.rnd_seed = kwargs['rnd_seed']
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory_dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = MemoryBase(self.memory_size, self.device)
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.temp_future_batch_idx = []

        self.num_updates = 0
        self.future_num_updates = 0
        self.sample_num = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.gt_label = None
        self.test_records = []
        self.n_model_cls = []
        self.knowledge_loss_rate = []
        self.knowledge_gain_rate = []
        self.forgetting_time = []
        self.note = kwargs['note']
        
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        self.f_period = kwargs['f_period']
        self.f_next_time = 0
        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167}
        self.total_samples = num_samples[self.dataset]

        self.waiting_batch = []
        self.waiting_batch_idx = []
        self.initialize_future()

        self.total_flops = 0.0
        self.writer = SummaryWriter(f'tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}')
        self.store_pickle = kwargs['store_pickle']
        self.select_criterion = kwargs['select_criterion']
        self.knn_top_k = kwargs['knn_top_k']
        self.softmax_temperature = kwargs['softmax_temperature']
        self.knn_sigma = kwargs['knn_sigma']
        self.loss_criterion = kwargs['loss_criterion']
        self.distill_coeff = kwargs['distill_coeff']


    # Memory 새로 정의 (not MemoryBase)
    def initialize_future(self):
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
        self.num_learned_class = 0
        self.num_learning_class = 1
        
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True
            
        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()

    def balanced_replace_memory(self, sample, sample_num=None):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.memory.cls_dict[sample['klass']]] += 1
            cls_to_replace = np.random.choice(
                np.flatnonzero(np.array(label_frequency) == np.array(label_frequency).max()))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace, sample_num = sample_num)
        else:
            self.memory.replace_sample(sample, sample_num = sample_num)

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

        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(int(self.future_num_updates))
            for stored_sample in self.temp_future_batch:
                self.update_memory(sample, self.future_sample_num)
                self.future_sample_num += 1
                #self.update_memory(stored_sample)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        return 0

    def update_memory(self, sample):
        pass

    def get_whole_batch(self):
        self.memory_dataloader.load_batch(self.memory.whole_retrieval(), self.memory.cls_dict, self.memory.sample_nums)
        batch = self.memory_dataloader.get_batch()
        return batch

    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch()
        self.load_batch()
        return batch

    # stream 또는 memory를 활용해서 batch를 load해라
    # data loader에 batch를 전달해주는 함수
    def load_batch(self):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            self.dataloader.load_batch(self.waiting_batch[0], self.memory.cls_dict, self.waiting_batch_idx[0])
            del self.waiting_batch[0]
            del self.waiting_batch_idx[0]

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            memory_batch, memory_batch_idx = self.memory.retrieval(self.memory_batch_size)
            self.waiting_batch.append(self.temp_future_batch + memory_batch)
            self.waiting_batch_idx.append(self.temp_future_batch_idx + memory_batch_idx)
        
    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss, train_acc)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []

    def add_new_class(self, class_name):
        self.added = True
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def infer_whole_memory(self, fc_weight):
        feature_dict = {}
        with torch.no_grad():
            self.model.train()
            data = self.get_whole_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            _, features = self.model(x, get_feature=True)
            unique_y = torch.unique(y).tolist()
            for u_y in unique_y:
                indices = (y == u_y).nonzero(as_tuple=True)[0]
                if u_y not in feature_dict.keys():
                    feature_dict[u_y] = [torch.index_select(features, 0, indices)]
                else:
                    feature_dict[u_y].append(torch.index_select(features, 0, indices))
            self.check_neural_collapse(feature_dict, fc_weight)

    def save_features(self, feature_pickle_name, class_pickle_name):
        feature_dict = {}
        with torch.no_grad():
            self.model.train()
            data = self.get_whole_batch()
            batchsize = 512
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            new_x, new_y = my_segmentation_transforms(x, y, self.real_num_classes)
            new_x = new_x.to(self.device)
            new_y = new_y.to(self.device)
            
            features = []
            for i in range(len(self.memory) // batchsize + 1):
                _, feature = self.model(x[i * batchsize:min((i + 1) * batchsize, len(x))], get_feature=True)
                features.append(feature)
            
            #self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device)) for i in range(512 // batchsize))], dim=0) 
            #_, features = self.model(x, get_feature=True)
            features = torch.cat(features,dim=0)
            unique_y = torch.unique(y).tolist()
            for u_y in unique_y:
                indices = (y == u_y).nonzero(as_tuple=True)[0]
                if u_y not in feature_dict.keys():
                    feature_dict[u_y] = [torch.index_select(features, 0, indices)]
                else:
                    feature_dict[u_y].append(torch.index_select(features, 0, indices))

        #pickle_name = str(int(self.online_iter)) + '_dist_dict.pickle'
        
        with open(feature_pickle_name, 'wb') as f:
            print("feature cls length", len(feature_dict.keys()))
            pickle.dump(feature_dict, f, pickle.HIGHEST_PROTOCOL)
        
        with open(class_pickle_name, 'wb') as f:
            pickle.dump(self.cls_dict, f, pickle.HIGHEST_PROTOCOL)        


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
            logit, loss = self.model_forward(x,y, sample_nums)

            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            #self.total_flops += (len(y) * self.backward_flops)

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def before_model_update(self):
        pass

    def after_model_update(self):
        self.update_schedule()

    '''
    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y)

        self.total_flops += (len(y) * self.forward_flops)
        return logit, loss
    '''

    def model_forward(self, x, y, sample_nums, get_feature=False):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        #do_cutmix = False
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                if get_feature:
                    logit, feature = self.model(x, get_feature=True)
                else:
                    logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                if get_feature:
                    logit, feature = self.model(x, get_feature=True)
                else:
                    logit = self.model(x)
                loss = self.criterion(logit, y)

        #self.total_flops += (len(y) * self.forward_flops)

        if get_feature:
            return logit, loss, feature
        else:
            return logit, loss

    def report_training(self, sample_num, train_loss, train_acc):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc, cls_acc):
        print("cls_acc")
        print(cls_acc)
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

    def report_future_test(self, sample_num, avg_loss, avg_acc, cls_acc):
        print("future_cls_acc")
        print(cls_acc)
        #writer.add_scalar(f"test/loss", avg_loss, sample_num)
        #writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"Future Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | TFLOPs {self.total_flops/1000:.2f}"
        )
        for idx in range(self.num_future_class):
            acc = cls_acc[self.num_learned_class + idx]
            logger.info(
                f"Class_Acc | Sample # {sample_num} | cls{self.num_learned_class + idx} {acc:.4f}"
            )

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()


    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, cls_order, future_train_dict, data_time):
        if self.added:
            test_df = pd.DataFrame(test_list)
            exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
            print("exp_test_df", len(exp_test_df))
            test_dataset = ImageDataset(
                exp_test_df,
                dataset=self.dataset,
                transform=self.test_transform,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir
            )
            self.test_loader = DataLoader(
                test_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
            )
            
            if self.use_future_eval:
                if len(cls_order) - len(self.exposed_classes) <= self.num_future_class:
                    self.future_train_loader = None
                    self.future_test_loader = None
                    
                else:
                    future_test_cls = cls_order[len(self.exposed_classes) : len(self.exposed_classes) + self.num_future_class]
                    ### make future_train_loader ###
                    future_train_list = []
                    for test_cls in future_test_cls:
                        future_train_list.extend(future_train_dict[test_cls])
                    
                    future_train_dataset = ImageDataset(
                        pd.DataFrame(future_train_list),
                        dataset=self.dataset,
                        transform=self.future_train_transform,
                        cls_list=self.exposed_classes+future_test_cls,
                        data_dir=self.data_dir
                    )
                    self.future_train_loader = DataLoader(
                        future_train_dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        num_workers=n_worker,
                    )
                    
                    ### make future_test_loader ###
                    future_test_df = pd.DataFrame(test_list)
                    exp_future_test_df = future_test_df[future_test_df['klass'].isin(future_test_cls)]
                    print("exp_future_test_df", len(exp_future_test_df))
                    future_test_dataset = ImageDataset(
                        exp_future_test_df,
                        dataset=self.dataset,
                        transform=self.test_transform,
                        cls_list=self.exposed_classes + future_test_cls,
                        data_dir=self.data_dir
                    )
                    self.future_test_loader = DataLoader(
                        future_test_dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        num_workers=n_worker,
                    )

                    future_eval_dict = self.future_evaluation()
                    self.report_future_test(sample_num, future_eval_dict["avg_loss"], future_eval_dict["avg_acc"], future_eval_dict["cls_acc"])              
                    
        eval_dict = self.evaluation(self.test_loader, self.criterion)
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["cls_acc"])
        
        self.added = False
        return eval_dict


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        feature_dict = {}
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, features = self.model(x, get_feature=True)
                unique_y = torch.unique(y).tolist()
                for u_y in unique_y:
                    indices = (y == u_y).nonzero(as_tuple=True)[0]
                    if u_y not in feature_dict.keys():
                        feature_dict[u_y] = [torch.index_select(features, 0, indices)] 
                    else:
                        feature_dict[u_y].append(torch.index_select(features, 0, indices))

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        # self.check_neural_collapse(feature_dict)

        # neural collapse check at eval stage (start)
        '''
        for name, param in self.model.named_parameters():
            if "fc.weight" in name:
                fc_weight = copy.deepcopy(param)
        label_list = []
        feature_list = []
        for key in list(feature_dict.keys()):
            feature_dict[key] = torch.cat(feature_dict[key])
        print("**********")
        
        for key in list(feature_dict.keys()):
            std = torch.mean(torch.std(feature_dict[key], dim=0)).item()
            feature_list.extend(feature_dict[key])
            for _ in range(len(feature_dict[key])):
                label_list.append(key)
            dist = ((fc_weight[key]-torch.mean(feature_dict[key], dim=0))**2).sum().sqrt().item()
            if key not in self.std_dict.keys():
                self.std_dict[key] = [std]
                self.dist_dict[key] = [dist]
            else:
                self.std_dict[key].append(std)
                self.dist_dict[key].append(dist)
        print("**********")
        print()
        
        std_dict_pickle_name = str(int(self.online_iter)) + '_std_dict.pickle'
        dist_dict_pickle_name = str(int(self.online_iter)) + '_dist_dict.pickle'
        with open(std_dict_pickle_name, 'wb') as f:
            pickle.dump(self.std_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(dist_dict_pickle_name, 'wb') as f:
            pickle.dump(self.dist_dict, f, pickle.HIGHEST_PROTOCOL)
            
        label_list = np.array(label_list)
        color_list = ["violet", "limegreen", "orange","pink","blue","brown","red","grey","yellow","green"]
        tsne_model = TSNE(n_components=2)
        cluster = np.array(tsne_model.fit_transform(torch.stack(feature_list).cpu()))
        for i in range(len(list(feature_dict.keys()))):
            idx = np.where(np.array(label_list) == i)
            label = "class" + str(i)
            plt.scatter(cluster[idx[0], 0], cluster[idx[0], 1], marker='.', c=color_list[i], label=label)
        
        fig_name = str(int(self.online_iter)) + "_tsne_figure_" + str(self.sample_num) + ".png"
        plt.savefig(fig_name)
        '''
        # neural collapse check at eval stage (end)

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def check_neural_collapse(self, feature_dict, fc_weight):
        '''
        for name, param in self.model.named_parameters():
            if "fc.weight" in name:
                fc_weight = copy.deepcopy(param)
        '''
        label_list = []
        feature_list = []
        for key in list(feature_dict.keys()):
            feature_dict[key] = torch.cat(feature_dict[key])

        mean_features = []
        for key in list(feature_dict.keys()):
            mean_features.append(torch.mean(feature_dict[key], dim=0))
            std = torch.mean(torch.std(feature_dict[key], dim=0)).item()
            feature_list.extend(feature_dict[key])
            for _ in range(len(feature_dict[key])):
                label_list.append(key)
            dist = ((fc_weight[key]-torch.mean(feature_dict[key], dim=0))**2).sum().sqrt().item()
            if key not in self.std_dict.keys():
                self.std_dict[key] = [std]
                self.dist_dict[key] = [dist]
            else:
                self.std_dict[key].append(std)
                self.dist_dict[key].append(dist)
        
        # calculate between feature std calculate
        self.btw_std_dict.append(torch.mean(torch.std(torch.stack(mean_features), dim=0)).item())
        
        btw_std_dict_pickle_name = str(int(self.online_iter)) + '_btw_std_dict.pickle'        
        std_dict_pickle_name = str(int(self.online_iter)) + '_std_dict.pickle'
        dist_dict_pickle_name = str(int(self.online_iter)) + '_dist_dict.pickle'
        
        with open(btw_std_dict_pickle_name, 'wb') as f:
            pickle.dump(self.btw_std_dict, f, pickle.HIGHEST_PROTOCOL)
        
        with open(std_dict_pickle_name, 'wb') as f:
            pickle.dump(self.std_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(dist_dict_pickle_name, 'wb') as f:
            pickle.dump(self.dist_dict, f, pickle.HIGHEST_PROTOCOL)

        label_list = np.array(label_list)
        color_list = ["violet", "limegreen", "orange","pink","blue","brown","red","grey","yellow","green"] #cifar10 기준
        tsne_model = TSNE(n_components=2)
        cluster = np.array(tsne_model.fit_transform(torch.stack(feature_list).cpu()))
        plt.figure()
        for i in range(len(list(feature_dict.keys()))):
            idx = np.where(np.array(label_list) == i)
            label = "class" + str(i)
            plt.scatter(cluster[idx[0], 0], cluster[idx[0], 1], marker='.', c=color_list[i], label=label)

        fig_name = str(int(self.online_iter)) + "_tsne_figure_" + str(self.sample_num) + ".png"
        plt.savefig(fig_name)
        

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

    def _interpret_pred(self, y, pred):
        ret_num_data = torch.zeros(self.n_classes).to(self.device)
        ret_corrects = torch.zeros(self.n_classes).to(self.device)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects


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
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred.detach().cpu().numpy())
                gts.append(y.detach().cpu().numpy())
        preds = np.concatenate(preds)
        if self.gt_label is None:
            gts = np.concatenate(gts)
            self.gt_label = gts
        self.test_records.append(preds)
        self.n_model_cls.append(copy.deepcopy(self.num_learned_class))
        '''
        if len(self.test_records) > 1:
            klr, kgr, = self.calculate_online_forgetting(self.n_classes, self.gt_label, self.test_records[-2], self.test_records[-1], self.n_model_cls[-2], self.n_model_cls[-1])
            self.knowledge_loss_rate.append(klr)
            self.knowledge_gain_rate.append(kgr)
            self.forgetting_time.append(sample_num)
            logger.info(f'KLR {klr} | KGR {kgr}')
            np.save(self.save_path + '_KLR.npy', self.knowledge_loss_rate)
            np.save(self.save_path + '_KGR.npy', self.knowledge_gain_rate)
            np.save(self.save_path + '_forgetting_time.npy', self.forgetting_time)
        '''


    def calculate_online_forgetting(self, n_classes, y_gt, y_t1, y_t2, n_cls_t1, n_cls_t2):
        total_cnt = len(y_gt)
        cnt_gt = np.zeros(n_classes)
        cnt_y1 = np.zeros(n_cls_t1)
        cnt_y2 = np.zeros(n_cls_t2)
        correct_y1 = np.zeros(n_classes)
        correct_y2 = np.zeros(n_classes)
        correct_both = np.zeros(n_classes)
        for i, gt in enumerate(y_gt):
            y1, y2 = y_t1[i], y_t2[i]
            cnt_gt[gt] += 1
            cnt_y1[y1] += 1
            cnt_y2[y2] += 1
            if y1 == gt:
                correct_y1[gt] += 1
                if y2 == gt:
                    correct_y2[gt] += 1
                    correct_both[gt] += 1
            elif y2 == gt:
                correct_y2[gt] += 1

        gt_prob = cnt_gt/total_cnt
        y1_prob = cnt_y1/total_cnt
        y2_prob = cnt_y2/total_cnt

        probs = np.zeros([n_classes, n_cls_t1, n_cls_t2])

        for i in range(n_classes):
            cls_prob = gt_prob[i]
            notlearned_prob = 1 - (correct_y1[i] + correct_y2[i] - correct_both[i])/cnt_gt[i]
            forgotten_prob = (correct_y1[i] - correct_both[i]) / cnt_gt[i]
            newlearned_prob = (correct_y2[i] - correct_both[i]) / cnt_gt[i]
            if i < n_cls_t1:
                marginal_y1 = y1_prob/(1-y1_prob[i])
                marginal_y1[i] = forgotten_prob/(notlearned_prob+1e-10)
            else:
                marginal_y1 = y1_prob
            if i < n_cls_t2:
                marginal_y2 = y2_prob/(1-y2_prob[i])
                marginal_y2[i] = newlearned_prob/(notlearned_prob+1e-10)
            else:
                marginal_y2 = y2_prob
            probs[i] = np.expand_dims(marginal_y1, 1) * np.expand_dims(marginal_y2, 0) * notlearned_prob * cls_prob
            if i < n_cls_t1 and i < n_cls_t2:
                probs[i][i][i] = correct_both[i]/total_cnt

        knowledge_loss = np.sum(probs*np.log(np.sum(probs, axis=(0, 1), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        knowledge_gain = np.sum(probs*np.log(np.sum(probs, axis=(0, 2), keepdims=True) * probs / (np.sum(probs, axis=0, keepdims=True)+1e-10) / (np.sum(probs, axis=2, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        prob_gt_y1 = probs.sum(axis=2)
        prev_total_knowledge = np.sum(prob_gt_y1*np.log(prob_gt_y1/(np.sum(prob_gt_y1, axis=0, keepdims=True)+1e-10)/(np.sum(prob_gt_y1, axis=1, keepdims=True)+1e-10)+1e-10))/np.log(n_classes)
        max_knowledge = np.log(n_cls_t2)/np.log(n_classes)

        knowledge_loss_rate = knowledge_loss/prev_total_knowledge
        knowledge_gain_rate = knowledge_gain/(max_knowledge-prev_total_knowledge)
        return knowledge_loss_rate, knowledge_gain_rate

    def future_evaluation(self):
        
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        
        # k-shot training temp_model using future data
        temp_model = copy.deepcopy(self.model)
        temp_model.train()
        for name, param in temp_model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        
        for i in range(self.future_training_iterations):
            for i, data in enumerate(self.future_train_loader):
                x = data["image"].to(self.device)
                y = data["label"].to(self.device)
                
                self.optimizer.zero_grad()

                # logit can not be used anymore
                with torch.cuda.amp.autocast(self.use_amp):
                    logit, feature = temp_model(x, get_feature=True)
                    loss = self.criterion(logit, y)
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

        # for calculating forward transfer
        for i, data in enumerate(self.future_test_loader):
            x = data["image"]
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)

            logit, _ = temp_model(x)

            loss = self.criterion(logit, y)
            pred = torch.argmax(logit, dim=-1)
            _, preds = logit.topk(self.topk, 1, True, True)

            total_correct += torch.sum(preds == y.unsqueeze(1)).item()
            total_num_data += y.size(0)

            xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
            correct_l += correct_xlabel_cnt.detach().cpu()
            num_data_l += xlabel_cnt.detach().cpu()

            total_loss += loss.item()
            label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(self.future_test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        
        return ret

    def get_flops_parameter(self):
        _, _, _, inp_size, inp_channel = get_statistics(dataset=self.dataset)
        [forward_mac, backward_mac, params, fc_params, buffers], \
            [initial_block_forward_mac, initial_block_backward_mac, initial_block_params], \
            [group1_block0_forward_mac, group1_block0_backward_mac, group1_block0_params], \
            [group1_block1_forward_mac, group1_block1_backward_mac, group1_block1_params], \
            [group2_block0_forward_mac, group2_block0_backward_mac, group2_block0_params], \
            [group2_block1_forward_mac, group2_block1_backward_mac, group2_block1_params], \
            [group3_block0_forward_mac, group3_block0_backward_mac, group3_block0_params], \
            [group3_block1_forward_mac, group3_block1_backward_mac, group3_block1_params], \
            [group4_block0_forward_mac, group4_block0_backward_mac, group4_block0_params], \
            [group4_block1_forward_mac, group4_block1_backward_mac, group4_block1_params], \
            [fc_forward_mac, fc_backward_mac, _] = get_model_complexity_info(self.model,
                                                                             (inp_channel, inp_size, inp_size),
                                                                             as_strings=False,
                                                                             print_per_layer_stat=False, verbose=True,
                                                                             criterion=self.criterion,
                                                                             original_opt=self.optimizer,
                                                                             opt_name=self.opt_name, lr=self.lr)

        # flops = float(mac) * 2 # mac은 string 형태
        print("forward mac", forward_mac, "backward mac", backward_mac, "params", params, "fc_params", fc_params,
              "buffers", buffers)
        print("initial forward mac", initial_block_forward_mac, "initial backward mac", initial_block_backward_mac,
              "initial params", initial_block_params)
        print("group1 block0 forward mac", group1_block0_forward_mac, "group1 block0 backward mac",
              group1_block0_backward_mac, "group1 block0 params", group1_block0_params)
        print("group1 block1 forward mac", group1_block1_forward_mac, "group1 block1 backward mac",
              group1_block1_backward_mac, "group1 block1 params", group1_block1_params)
        print("group2 block0 forward mac", group2_block0_forward_mac, "group2 block0 backward mac",
              group2_block0_backward_mac, "group2 block0 params", group2_block0_params)
        print("group2 block1 forward mac", group2_block1_forward_mac, "group2 block1 backward mac",
              group2_block1_backward_mac, "group2 block1 params", group2_block1_params)
        print("group3 block0 forward mac", group3_block0_forward_mac, "group3 block0 backward mac",
              group3_block0_backward_mac, "group3 block0 params", group3_block0_params)
        print("group3 block1 forward mac", group3_block1_forward_mac, "group3 block1 backward mac",
              group3_block1_backward_mac, "group3 block1 params", group3_block1_params)
        print("group4 block0 forward mac", group4_block0_forward_mac, "group4 block0 backward mac",
              group4_block0_backward_mac, "group4 block0 params", group4_block0_params)
        print("group4 block1 forward mac", group4_block1_forward_mac, "group4 block1 backward mac",
              group4_block1_backward_mac, "group4 block1 params", group4_block1_params)

        print("fc forward mac", fc_forward_mac, "fc backward mac", fc_backward_mac, "fc params", fc_params)

        self.forward_flops = forward_mac / 10e9
        self.initial_forward_flops = initial_block_forward_mac / 10e9
        self.group1_block0_forward_flops = group1_block0_forward_mac / 10e9
        self.group2_block0_forward_flops = group2_block0_forward_mac / 10e9
        self.group3_block0_forward_flops = group3_block0_forward_mac / 10e9
        self.group4_block0_forward_flops = group4_block0_forward_mac / 10e9
        self.group1_block1_forward_flops = group1_block1_forward_mac / 10e9
        self.group2_block1_forward_flops = group2_block1_forward_mac / 10e9
        self.group3_block1_forward_flops = group3_block1_forward_mac / 10e9
        self.group4_block1_forward_flops = group4_block1_forward_mac / 10e9
        self.fc_forward_flops = fc_forward_mac / 10e9

        self.backward_flops = backward_mac / 10e9
        self.initial_backward_flops = initial_block_backward_mac / 10e9
        self.group1_block0_backward_flops = group1_block0_backward_mac / 10e9
        self.group2_block0_backward_flops = group2_block0_backward_mac / 10e9
        self.group3_block0_backward_flops = group3_block0_backward_mac / 10e9
        self.group4_block0_backward_flops = group4_block0_backward_mac / 10e9
        self.group1_block1_backward_flops = group1_block1_backward_mac / 10e9
        self.group2_block1_backward_flops = group2_block1_backward_mac / 10e9
        self.group3_block1_backward_flops = group3_block1_backward_mac / 10e9
        self.group4_block1_backward_flops = group4_block1_backward_mac / 10e9
        self.fc_backward_flops = fc_backward_mac / 10e9

        self.comp_backward_flops = [
            self.initial_backward_flops,
            self.group1_block0_backward_flops, self.group1_block1_backward_flops,
            self.group2_block0_backward_flops, self.group2_block1_backward_flops,
            self.group3_block0_backward_flops, self.group3_block1_backward_flops,
            self.group4_block0_backward_flops, self.group4_block1_backward_flops,
            self.fc_backward_flops
        ]

        self.params = params / 10e9
        self.fc_params = fc_params / 10e9
        self.buffers = buffers / 10e9


class MemoryBase:
    def __init__(self, memory_size, device, ood_strategy=None):
        self.memory_size = memory_size
        self.images = []
        self.labels = []
        self.sample_nums = []
        self.update_buffer = ()
        self.cls_dict = dict()
        self.cls_list = []
        self.cls_count = []
        self.cls_idx = []
        self.usage_count = np.array([])
        self.class_usage_count = np.array([])
        self.current_images = []
        self.current_labels = []
        self.current_cls_count = [0 for _ in self.cls_list]
        self.current_cls_idx = [[] for _ in self.cls_list]
        self.ood_strategy = ood_strategy
        self.device = device

    def __len__(self):
        return len(self.images)

    def replace_sample(self, sample, idx=None, sample_num=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            assert len(self.images) < self.memory_size
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.images.append(sample)
            self.labels.append(self.cls_dict[sample['klass']])
            if sample_num is not None:
                self.sample_nums.append(sample_num)
        else:
            assert idx < self.memory_size
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.images[idx] = sample
            self.labels[idx] = self.cls_dict[sample['klass']]
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            if sample_num is not None:
                self.sample_nums[idx] = sample_num
                
    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.cls_list)
        self.cls_list.append(class_name)
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.class_usage_count = np.append(self.class_usage_count, 0.0)
        print("!!added", class_name)
        print("self.cls_dict", self.cls_dict)


    def whole_retrieval(self):
        memory_batch = []
        indices = list(range(len(self.images)))
        for i in indices:
            memory_batch.append(self.images[i])
        return memory_batch

    def generate_ood_class(self, num_samples=1):
        if self.ood_strategy == "cutmix":
            if len(self.cls_idx) <= 1:
                return None
            pair = np.random.choice(len(self.cls_idx), 2, replace=False)
            pair.sort()
            pair = tuple(pair)
            num_samples = min([len(self.cls_idx[pair[0]]), len(self.cls_idx[pair[1]]), num_samples])
            index_1 = np.random.choice(self.cls_idx[pair[0]], num_samples)
            index_2 = np.random.choice(self.cls_idx[pair[1]], num_samples)

            x1, y1 = np.array(self.images)[index_1], np.array(self.labels)[index_1]
            x2, y2 = np.array(self.images)[index_2], np.array(self.labels)[index_2]    
            return (pair, x1, x2)
        
        elif self.ood_strategy == "rotate":
            index = np.random.choice(range(len(self.images)), min(num_samples, len(self.images)), replace=False)
            x = np.array(self.images)[index]
            y = np.array(self.labels)[index]
            '''
            cls = np.random.choice(len(self.cls_idx), 1)
            num_samples = min([len(self.cls_idx[cls[0]]), num_samples])
            index = np.random.choice(self.cls_idx[cls[0]], num_samples)
            x = np.array(self.images)[index]
            '''
            return (x, y)
        
        #elif self.ood_strategy == "rotate":
            

    def retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        memory_batch_idx = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
            memory_batch_idx.append(self.sample_nums[i])
        return memory_batch, memory_batch_idx
