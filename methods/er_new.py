# When we make a new one, we should inherit the Finetune class.
import logging
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.cl_manager import CLManagerBase

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class ER(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        self.update_count = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)

    def update_memory(self, sample, sample_num=None):
        self.reservoir_memory(sample, sample_num)
        
    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.update_count += 1
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
            
        print("# of update:", self.update_count)
        print("# of data:", len(y))

        return total_loss / iterations, correct / num_data
    
    def reservoir_memory(self, sample, sample_num):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j, sample_num=sample_num)
        else:
            self.memory.replace_sample(sample, sample_num=sample_num)

