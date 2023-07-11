################################
# This code is referred by
# https://github.com/RaptorMai/online-continual-learning
################################

import os
import PIL
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from methods.er_new import ER
from torchvision import transforms
from methods.cl_manager import MemoryBase
from utils.data_loader import MultiProcessLoader
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

class SCR(ER):
    
    def __init__(
        self, train_datalist, test_datalist, device, **kwargs
    ):
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )
        
        self.memory.memory_size = kwargs["memory_size"]
        self.criterion = SupConLoss()
        self.transform = nn.Sequential(
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        )
        
    
    def initialize_future(self):
        self.cpu_transform = None
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = SCRMemory(self.data_dir, self.train_transform, self.memory_size, self.device)
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
        for _ in range(self.future_steps):
            self.load_batch()

        
        
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        torch.autograd.set_detect_anomaly(True)

        for _ in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            mem_x, _, mem_y = self.memory.retrieval(data["image"].shape[0])
            
            mem_x = torch.stack(tuple([mem_x[i]['tensor'] for i in range(len(mem_x))])).to(self.device)
            mem_y = torch.tensor(mem_y).to(self.device)
            combined_batch = torch.cat((mem_x, x))
            combined_labels = torch.cat((mem_y, y))
            combined_batch_aug = self.transform(combined_batch)
            
            #self.before_model_update()

            #logit, loss = self.model_forward(x,y, sample_nums)
            #_, preds = logit.topk(self.topk, 1, True, True)
            features = torch.cat([F.normalize(self.model(combined_batch, get_feature=True)[1], dim=1).unsqueeze(1), F.normalize(self.model(combined_batch_aug, get_feature=True)[1], dim=1).unsqueeze(1)], dim=1)
            loss = self.criterion(features, combined_labels)
            self.optimizer.zero_grad()

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
            #correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, 0
    
    
    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        self.model.eval()
        
        exemplar_means = {}
        cls_exemplar = {cls: [] for cls in self.memory.cls_dict.values()}
        buffer_filled = sum(self.memory.cls_count)
        for x, y in zip(self.memory.images[:buffer_filled], self.memory.labels[:buffer_filled]):
            cls_exemplar[y].append(x["tensor"])
        for cls, exemplar in cls_exemplar.items():
            features = []
            for ex in exemplar:
                ex = ex.to(self.device)
                feature = self.model(ex.unsqueeze(0), get_feature=True)[1].detach().clone()
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()  # Normalize
                features.append(feature)
            if len(features) == 0:
                pass
            else:
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()
            exemplar_means[cls] = mu_y
        
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                _, feature = self.model(x, get_feature=True)
                for j in range(feature.size(0)):
                    feature.data[j] = feature.data[j] / feature.data[j].norm()
                feature = feature.unsqueeze(2)
                means = torch.stack([exemplar_means[cls] for cls in self.memory.cls_dict.values()])
                
                means = torch.stack([means] * x.size(0))
                means = means.transpose(1, 2)
                feature = feature.expand_as(means)
                dists = (feature - means).pow(2).sum(1)
                _, pred_label = dists.min(1)
                correct_cnt = (np.array(list(self.memory.cls_dict.values()))[
                                   pred_label.tolist()] == y.cpu().numpy()).sum().item()
                
                total_correct += correct_cnt
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred_label)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()

        avg_acc = total_correct / total_num_data
        #avg_loss = total_loss / len(test_loader)
        avg_loss = 0
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret
    


class SCRMemory(MemoryBase):
    def __init__(self, data_dir, transform, memory_size, device, ood_strategy=None):
        super().__init__(
            memory_size, device, ood_strategy
        )
        self.data_dir = data_dir
        self.transform = transform
    
    def replace_sample(self, sample, idx=None, sample_num=None):
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        sample['tensor'] = self.transform(PIL.Image.open(os.path.join(self.data_dir, sample['file_name'])))
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
                
    def retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        memory_batch_idx = []
        memory_label = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
            memory_batch_idx.append(self.sample_nums[i])
            memory_label.append(self.labels[i])
        return memory_batch, memory_batch_idx, memory_label
    

            
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # mask (16, 16) : all 1
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
