import torch
import random
import os
import sys
import queue

import PIL
import numpy as np
from utils.augment import DataAugmentation, Preprocess, get_statistics

from PIL import ImageFilter
from torchvision import transforms
import kornia.augmentation as K

IS_WINDOWS = sys.platform == "win32"
TIMEOUT = 5.0

# from PyTorch Official Code
if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog:
        def __init__(self):
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
else:
    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead

def load_batch(samples, data_dir, transform=None):
    return torch.stack([load_data(sample, data_dir, transform) for sample in samples])

def load_data(sample, data_dir, transform=None):
    img_name = sample["file_name"]
    img_path = os.path.join(data_dir, img_name)
    image = PIL.Image.open(img_path).convert("RGB")
    if transform:
        image = transform(image)
    return image

    
@torch.no_grad()
def worker_loop(index_queue, data_queue, data_dir, transform, transform_on_gpu=False, cpu_transform=None, device='cpu', use_kornia=False, transform_on_worker=True, test_transform=None):
    watchdog = ManagerWatchdog()
    if use_kornia:
        if 'cifar100' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar100')
        elif 'cifar10' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
        elif 'tinyimagenet' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='tinyimagenet')
        elif 'imagenet' in data_dir:
            mean, std, n_classes, inp_size, _ = get_statistics(dataset='imagenet')
        preprocess = Preprocess(inp_size)
        kornia_randaug = DataAugmentation(inp_size, mean, std)
    
    
    if 'cifar100' in data_dir:
        mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar100')
    elif 'cifar10' in data_dir:
        mean, std, n_classes, inp_size, _ = get_statistics(dataset='cifar10')
    elif 'tinyimagenet' in data_dir:
        mean, std, n_classes, inp_size, _ = get_statistics(dataset='tinyimagenet')
    elif 'imagenet' in data_dir:
        mean, std, n_classes, inp_size, _ = get_statistics(dataset='imagenet')
        
    cpu_transform = transforms.Compose([
        transforms.RandAugment(),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.ToTensor(),
    ])

    def kornia_transform(images):
        images = K.RandomResizedCrop((inp_size, inp_size), scale=(0.2, 1.0))(images)
        images = K.RandomGrayscale(p=0.2)(images)
        images = K.RandomHorizontalFlip()(images)
        return images
    
    gpu_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean, std),
    ])
    
    while watchdog.is_alive():
        try:
            r = index_queue.get(timeout=TIMEOUT)
        except queue.Empty:
            continue
        
        data = dict()
        images = []
        labels = []
        indexs = []
        test_images = []
        
        # twc
        task_ids = []
        if len(r) > 0:
            for sample in r:
                if use_kornia:
                    img_name = sample["file_name"]
                    img_path = os.path.join(data_dir, img_name)
                    image = PIL.Image.open(img_path).convert("RGB")
                    images.append(preprocess(image))
                elif transform_on_gpu:
                    images.append(load_data(sample, data_dir, cpu_transform))
                else:
                    images.append(load_data(sample, data_dir, transform))
                    if test_transform is not None:
                        test_images.append(load_data(sample, data_dir, test_transform))
                labels.append(sample["label"])
                indexs.append(sample["sample_num"])
                if 'task_id' in sample:
                    task_ids.append(sample["task_id"])
                    
            if transform_on_worker:
                if use_kornia:
                    images = kornia_randaug(torch.stack(images).to(device))
                elif transform_on_gpu:
                    if test_transform is not None:
                        test_images = test_transform(torch.stack(images).float().to(device))
                    # images = transform(torch.stack(images).to(device))
                    images = torch.stack(images).to(device)
                    images = kornia_transform(images)
                    images = gpu_transform(images)
                else:
                    images = torch.stack(images)
                    if test_transform is not None:
                        test_images = torch.stack(test_images)
            data['image'] = images
            data['test_image'] = test_images
            data['label'] = torch.LongTensor(labels)
            data['sample_num'] = torch.LongTensor(indexs)
            if len(task_ids) > 0:
                data['task_id'] = torch.LongTensor(task_ids)
            data_queue.put(data)
        else:
            data_queue.put(None)



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
