import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torch
import pandas as pd
from models import mnist, cifar, imagenet
from torch.utils.data import DataLoader
from onedrivedownloader import download as dn
from torch.optim import SGD
from numbers import Number
import numpy as np
from utils.data_loader import get_train_datalist, ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, get_test_datalist
import torch.nn.functional as F
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
from utils.my_augment import Kornia_Randaugment
from torchvision import transforms
from tqdm import tqdm
from models.cifar import resnet18
from models.cifar_moco import resnet18_moco

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        """Module to calculate the accuracy.

        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super().__init__()
        self.topk = topk

    def forward(self, pred, target, real_entered_num_class = None, real_num_class = None):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, real_entered_num_class = real_entered_num_class, real_num_class = real_num_class)

class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (inp_size,inp_size)),
            K.RandomCrop(size = (inp_size,inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
            )
        #self.cutmix = K.RandomCutMix(p=0.5)

    def set_cls_magnitude(self, option, current_cls_loss, class_count):
        self.randaugmentation.set_cls_magnitude(option, current_cls_loss, class_count)

    def get_cls_magnitude(self):
        return self.randaugmentation.get_cls_magnitude()

    def get_cls_num_ops(self):
        return self.randaugmentation.get_cls_num_ops()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, labels=None) -> Tensor:
        #if labels is None or len(self.randaugmentation.cls_num_ops) == 0:
        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (self.inp_size, self.inp_size)),
            K.RandomCrop(size = (self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
            )
        #print("transform")
        #print(self.transforms)
        x_out = self.transforms(x)  # BxCxHxW
        '''
        else:
            additional_aug = self.randaugmentation.form_transforms(list(set((labels))))
            
            self.before_transforms = nn.Sequential(
                K.Resize(size = (self.inp_size, self.inp_size)),
                K.RandomCrop(size = (self.inp_size, self.inp_size)),
                K.RandomHorizontalFlip(p=1.0)
                )
            x_out = self.before_transforms(x)
            
            for i in range(len(x)):
                add_transform = nn.Sequential(*additional_aug[labels[i]])
                x_out[i] = add_transform(x_out[i])
                
            self.after_transforms = nn.Sequential(
                K.Normalize(self.mean, self.std)
                )
            x_out = self.transforms(x_out)  # BxCxHxW
        '''
        ##### check transforms
        # print("self.transform")
        # print(self.transforms)

        #x_out, _ = self.cutmix(x_out)
        return x_out


def get_transform(dataset, transform_list, gpu_transform, use_kornia=True):
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset)
    if use_kornia:
        train_transform = DataAugmentation(inp_size, mean, std)
    else:
        train_transform = []
        if "cutout" in transform_list:
            train_transform.append(Cutout(size=16))
            if gpu_transform:
                gpu_transform = False
                print("cutout not supported on GPU!")
        if "randaug" in transform_list:
            train_transform.append(transforms.RandAugment())
            
        if "autoaug" in transform_list:
            if hasattr(transform_list, 'AutoAugment'):
                if 'cifar' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
                elif 'imagenet' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            else:
                train_transform.append(select_autoaugment(dataset))
                gpu_transform = False
        if "trivaug" in transform_list:
            train_transform.append(transforms.TrivialAugmentWide())
        if gpu_transform:
            train_transform = transforms.Compose([
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    print(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform

def select_optimizer(opt_name, lr, model):
    if "adam" in opt_name:
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.Adam(params, lr=lr, weight_decay=0)
    elif "sgd" in opt_name:
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.SGD(
            params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    if 'freeze_fc' not in opt_name:
        opt.add_param_group({'params': model.fc.parameters()})
    return opt

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class DR_loss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            pure_num=None,
            augmented_num=None,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        
        if self.reduction == "mean":
            if augmented_num is None:
                loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)
            else:
                loss = ((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2
                loss = 0.5 * ((torch.mean(loss[:pure_num]) + torch.mean(loss[pure_num:])) / 2)
                
        elif self.reduction == "none":
            loss = 0.5 * (((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight

class DR_Reverse_loss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=0.01,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            targets,
            num_classes,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        #dot = torch.sum(feat * targets, dim=1)
        dot = torch.sum(feat @ targets.T, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        
        if self.reduction == "mean":
            loss = 0.5 * torch.mean((dot ** 2) / h_norm2)
        elif self.reduction == "none":
            loss = 0.5 * (((dot) ** 2) / h_norm2)

        return loss * self.loss_weight * (1 / num_classes)

def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1, ), thrs=0., real_entered_num_class=None, real_num_class=None):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')
    if real_entered_num_class is not None:
        inf_index = []
        for i in range(4):
            #inf_index += list(range(i * real_num_class, i * real_num_class + real_entered_num_class))
            inf_index += list(range(i * real_num_class + real_entered_num_class, min((i+1) * real_num_class, pred.shape[1])))
        pred[:, inf_index] = float('-inf')
        
    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred = pred.float()
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()

    '''
    print("target")
    print(target.view(1, -1).expand_as(pred_label))
    print("pred_label")
    print(pred_label)
    '''
    
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    correct_count = int(torch.sum(correct).item())
    return res, correct_count


def accuracy(pred, target, topk=1, thrs=0., real_entered_num_class = None, real_num_class = None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
                 if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res, correct_count = accuracy_torch(pred, target, topk, thrs, real_entered_num_class, real_num_class)
    return (res[0], correct_count) if return_single else (res, correct_count)

'''
def DR_loss(feat, target, loss_weight):
    dot = torch.sum(feat * target, dim=1)
    h_norm2 = torch.ones_like(dot)
    m_norm2 = torch.ones_like(dot)
    loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

    return loss * loss_weight
'''

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler



def get_ckpt_remote_url(pre_dataset):
    if pre_dataset == "cifar100":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs18_cifar100.pth"

    elif pre_dataset == "tinyimgR":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok" width="98" height="120" frameborder="0" scrolling="no"></iframe>', "erace_pret_on_tinyr.pth"

    elif pre_dataset == "imagenet":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs50_imagenet_full.pth"

    else:
        raise ValueError("Unknown auxiliary dataset")


def load_initial_checkpoint(pre_dataset, model, device, load_cp_path = None):
    url, ckpt_name = get_ckpt_remote_url(pre_dataset)
    load_cp_path = load_cp_path if load_cp_path is not None else './checkpoints/'
    print("Downloading checkpoint file...")
    dn(url, load_cp_path)
    print(f"Downloaded in: {load_cp}")
    net = load_cp(load_cp_path, model, device, moco=True)
    print("Loaded!")
    return net

def generate_initial_checkpoint(net, pre_dataset, pre_epochs, num_aux_classes, device, opt_args):
    aux_dset, aux_test_dset = get_aux_dataset()
    net.fc = torch.nn.Linear(net.fc.in_features, num_aux_classes).to(device)
    net.train()
    opt = SGD(net.parameters(), lr=opt_args["lr"], weight_decay=opt_args["optim_wd"], momentum=opt_args["optim_mom"])
    sched = None
    if self.args.pre_dataset.startswith('cub'):
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[80, 150, 250], gamma=0.5)
    elif 'tinyimg' in self.args.pre_dataset.lower():
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[20, 30, 40, 45], gamma=0.5)

    for e in range(pre_epochs):
        for i, (x, y, _) in tqdm(enumerate(aux_dl), desc='Pre-training epoch {}'.format(e), leave=False, total=len(aux_dl)):
            y = y.long()
            opt.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            aux_out = net(x)
            aux_loss = loss(aux_out, y)
            aux_loss.backward()
            opt.step()

        if sched is not None:
            sched.step()
        if e % 5 == 4:
            print(e, f"{self.mini_eval()*100:.2f}%")
    from datetime import datetime
    # savwe the model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelpath = "my_checkpoint" + '_' + now + '.pth'
    torch.save(net.state_dict(), modelpath)
    print(modelpath)


def load_cp(cp_path, net, device, moco=False) -> None:
    """
    Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

    :param cp_path: path to checkpoint
    :param new_classes: ignore and rebuild classifier with size `new_classes`
    :param moco: if True, allow load checkpoint for Moco pretraining
    """
    print("net")
    print([name for name, _ in net.named_parameters()])
    s = torch.load(cp_path, map_location=device)
    print("s keys", s.keys())
    '''
    if 'state_dict' in s:  # loading moco checkpoint
        if not moco:
            raise Exception(
                'ERROR: Trying to load a Moco checkpoint without setting moco=True')
        s = {k.replace('encoder_q.', ''): i for k,
             i in s['state_dict'].items() if 'encoder_q' in k}
    '''

    #if not ignore_classifier: # online CL이므로 fc out-dim을 1부터 시작
    net.fc = torch.nn.Linear(
        net.fc.in_features, 1).to(device) # online이므로 num_aux_classes => 1

    for k in list(s):
        if 'fc' in k:
            s.pop(k)
    for k in list(s):
        if 'net' in k:
            s[k[4:]] = s.pop(k)
    for k in list(s):
        if 'wrappee.' in k:
            s[k.replace('wrappee.', '')] = s.pop(k)
    for k in list(s):
        if '_features' in k:
            s.pop(k)

    try:
        net.load_state_dict(s)
    except:
        _, unm = net.load_state_dict(s, strict=False)
        print("unm")
        print(unm)
        '''
        if new_classes is not None or ignore_classifier:
            assert all(['classifier' in k for k in unm]
                       ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None, f"Missing keys: {unm}"
        '''

    return net
'''
def partial_distill_loss(model, net_partial_features: list, pret_partial_features: list,
                         targets, teacher_forcing: list = None, extern_attention_maps: list = None):

    assert len(net_partial_features) == len(
        pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

    if teacher_forcing is None or extern_attention_maps is None:
        assert teacher_forcing is None
        assert extern_attention_maps is None

    loss = 0
    attention_maps = []

    for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
        assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

        adapter = getattr(
            model, f"adapter_{i+1}")

        pret_feat = pret_feat.detach()

        if teacher_forcing is None:
            curr_teacher_forcing = torch.zeros(
                len(net_feat,)).bool().to(self.device)
            curr_ext_attention_map = torch.ones(
                (len(net_feat), adapter.c)).to(self.device)
        else:
            curr_teacher_forcing = teacher_forcing
            curr_ext_attention_map = torch.stack(
                [b[i] for b in extern_attention_maps], dim=0).float()

        adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                              teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

        loss += adapt_loss
        attention_maps.append(adapt_attention.detach().cpu().clone().data)

    return loss / (i + 1), attention_maps
'''
def get_data_loader(opt_dict, dataset, pre_train=False):
    if pre_train:
        batch_size = 128
    else:
        batch_size = opt_dict['batchsize']

    # pre_dataset을 위한 dataset 불러오고 dataloader 생성
    train_transform, test_transform = get_transform(dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

    test_datalist = get_test_datalist(dataset)
    train_datalist, cls_dict, cls_addition = get_train_datalist(dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

    # for debugging!
    # train_datalist = train_datalist[:2000]

    exp_train_df = pd.DataFrame(train_datalist)
    exp_test_df = pd.DataFrame(test_datalist)

    train_dataset = ImageDataset(
        exp_train_df,
        dataset=dataset,
        transform=train_transform,
        preload = True,
        use_kornia=True,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=opt_dict["n_worker"],
    )
    test_dataset = ImageDataset(
        exp_test_df,
        dataset=dataset,
        transform=test_transform,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,#opt_dict["batchsize"],
        num_workers=opt_dict["n_worker"],
    )

    return train_loader, test_loader

def select_moco_model(model_name, dataset, num_classes=None, opt_dict=None, pre_trained=False, G=False, F=False, Neck = False, K = 4096, dim=182):
    return resnet18_moco(pretrained=pre_trained, dataset=False, progress=True, F=False, G=False, Neck = Neck, K=K, dim=dim)

def select_model(model_name, dataset, num_classes=None, opt_dict=None, pre_trained=False, G=False, F=False, Neck = False):
    return resnet18(pretrained=pre_trained, dataset=False, progress=True, F=F, G=G, Neck = Neck)

##### for ASER #####
def compute_knn_sv(model, eval_x, eval_y, cand_x, cand_y, k, device="cpu"):
    """
        Compute KNN SV of candidate data w.r.t. evaluation data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                eval_y (tensor): evaluation label tensor.
                cand_x (tensor): candidate data tensor.
                cand_y (tensor): candidate label tensor.
                k (int): number of nearest neighbours.
                device (str): device for tensor allocation.
            Returns
                sv_matrix (tensor): KNN Shapley value matrix of candidate data w.r.t. evaluation data.
    """
    # Compute KNN SV score for candidate samples w.r.t. evaluation samples
    n_eval = eval_x.size(0)
    n_cand = cand_x.size(0)
    # Initialize SV matrix to matrix of -1
    sv_matrix = torch.zeros((n_eval, n_cand), device=device)
    # Get deep features
    eval_df, cand_df = deep_features(model, eval_x, n_eval, cand_x, n_cand)
    # Sort indices based on distance in deep feature space
    sorted_ind_mat = sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

    # Evaluation set labels
    el = eval_y
    el_vec = el.repeat([n_cand, 1]).T
    # Sorted candidate set labels
    cl = cand_y[sorted_ind_mat]

    # Indicator function matrix
    indicator = (el_vec == cl).float()
    indicator_next = torch.zeros_like(indicator, device=device)
    indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
    indicator_diff = indicator - indicator_next

    cand_ind = torch.arange(n_cand, dtype=torch.float, device=device) + 1
    denom_factor = cand_ind.clone()
    denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
    numer_factor = cand_ind.clone()
    numer_factor[k:n_cand - 1] = k
    numer_factor[n_cand - 1] = 1
    factor = numer_factor / denom_factor

    indicator_factor = indicator_diff * factor
    indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

    # Row indices
    row_ind = torch.arange(n_eval, device=device)
    row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

    # Compute SV recursively
    sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum

    return sv_matrix


def deep_features(model, eval_x, n_eval, cand_x, n_cand):
    """
        Compute deep features of evaluation and candidate data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                n_eval (int): number of evaluation data.
                cand_x (tensor): candidate data tensor.
                n_cand (int): number of candidate data.
            Returns
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
    """
    # Get deep features
    if cand_x is None:
        num = n_eval
        total_x = eval_x
    else:
        num = n_eval + n_cand
        total_x = torch.cat((eval_x, cand_x), 0)

    # compute deep features with mini-batches
    total_x = maybe_cuda(total_x)
    deep_features_ = mini_batch_deep_features(model, total_x, num)

    eval_df = deep_features_[0:n_eval]
    cand_df = deep_features_[n_eval:]
    return eval_df, cand_df


def sorted_cand_ind(eval_df, cand_df, n_eval, n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
    cand_df_tile = cand_df.repeat([n_eval, 1])
    # Compute distance between evaluation and candidate feature vectors
    distance_vector = euclidean_distance(eval_df_repeat, cand_df_tile)
    # Turn distance vector into distance matrix
    distance_matrix = distance_vector.reshape((n_eval, n_cand))
    # Sort candidate set indices based on distance
    sorted_cand_ind_ = distance_matrix.argsort(1)
    return sorted_cand_ind_


# class ModelWrapper(nn.Module):
#     def __init__(self, model, output_layer_names, return_single=False):
#         super(ModelWrapper, self).__init__()
#         self.model = model
#         self.output_layer_names = output_layer_names
#         self.outputs = {}
#         self.return_single = return_single
#         add_hooks(self.model, self.outputs, self.output_layer_names)

#     def forward(self, images):
#         self.model(images)
#         output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
#         if self.return_single:
#             return output_vals[0]
#         else:
#             return output_vals



def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """

    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))



