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
        super().__init__(train_datalist, test_datalist, device, **kwargs)

    def update_memory(self, sample, sample_num=None):
        self.reservoir_memory(sample, sample_num)

    def reservoir_memory(self, sample, sample_num):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j, sample_num=sample_num)
        else:
            self.memory.replace_sample(sample, sample_num=sample_num)

