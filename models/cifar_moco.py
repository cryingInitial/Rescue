import torch
import torch.nn as nn
from collections import OrderedDict
from utils.torch_utils import load_state_dict_from_url 
# from mmcv.cnn import build_norm_layer

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

model_files = {
    'cifar10': '',
    'cifar100': 'cifar100_cls10_REMIND.pt',
    'tinyimagenet': '',
    'imagenet': '',
}

def safe_load_dict(model, new_model_state, should_resume_all_params=False):
    old_model_state = model.state_dict()
    c = 0
    if should_resume_all_params:
        for old_name, old_param in old_model_state.items():
            assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                old_name)
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'module':
            name = '.'.join(end)
        if name not in old_model_state:
            # print('%s not found in old model.' % name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

'''
class MLPFFNNeck(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.ln1 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels, in_channels * 2)),
            ('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
            ('relu', nn.LeakyReLU(0.1))
        ]))
        self.ln2 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels * 2, in_channels * 2)),
            ('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
            ('relu', nn.LeakyReLU(0.1))
        ]))
        self.ln3 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels * 2, out_channels, bias=False)),
        ]))
        if in_channels == out_channels:
            # self.ffn = nn.Identity()
            self.ffn = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(in_channels, out_channels, bias=False)),
            ]))
        else:
            self.ffn = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(in_channels, out_channels, bias=False)),
            ]))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        x = self.avg(inputs)
        x = x.view(inputs.size(0), -1)
        identity = x
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = x + self.ffn(identity)
        return x
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        # Save the pre-relu feature map for the attention module
        self.return_prerelu = False
        self.prerelu = None
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.return_prerelu:
            self.prerelu = out
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        # Save the pre-relu feature map for the attention module
        self.return_prerelu = False
        self.prerelu = None
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.return_prerelu:
            self.prerelu = out.clone()
        out = self.relu(out)

        return out

def _resnet(arch, block, layers, pretrained, progress, dataset=False, G=False, F=False, Neck = False, K=4096, dim=128, **kwargs):
    model = ResNet(block, layers, Neck = Neck, K = K, dim = dim, **kwargs)

    if pretrained:
        if G:
            model = ResNet_G(block, layers, **kwargs)
            resumed = torch.load(model_files[dataset])
            safe_load_dict(model, resumed['state_dict'], should_resume_all_params=True)
            return model
        elif F:
            model = ResNet_F(block, layers, **kwargs)
            resumed = torch.load(model_files[dataset])
            safe_load_dict(model, resumed['state_dict'], should_resume_all_params=True)
            return model
        
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 

        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def resnet18_moco(pretrained=False, progress=True, dataset=False, G=False, F=False, Neck=False, K=4096, dim=128, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, dataset=dataset, G=G, F=F, Neck = Neck, K=K, dim=dim, **kwargs)


class ResNet(nn.Module):

    def __init__(self, block, layers, Neck=False, K=4096, dim=128, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.return_prerelu = False
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        # neck layer forward or not
        self.neck_forward = Neck
        
        self.neck = MLPFFNNeck(in_channels = 512, out_channels=512)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # encoder define
        dim_mlp = self.fc.weight.shape[1]
        self.proj = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.fc
        )
        self.K = K
        self.dim = dim
        print("self.dim", self.dim)
        print("self.k", self.K)
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity
        
        # replace the keys at ptr (dequeue and enqueue)
        if ptr+batch_size >= self.K:
            self.queue[:, ptr : self.K] = keys.T[:,:self.K-ptr]
            keys = keys[self.K-ptr:]
            ptr = 0
            batch_size = keys.shape[0]
            
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, BasicBlock) or isinstance(c, Bottleneck):
                c.return_prerelu = enable
                
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        if self.return_prerelu:
            out0_pre = x.clone()
        x = self.relu(x)
        out0 = self.maxpool(x)

        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.avgpool(out4)
        
        if self.neck_forward:
            # neck layer
            feature = self.neck(out4)
        else:
            feature = out4

        feature = torch.flatten(feature, 1)
        
        # for moco 
        proj_output = self.proj(torch.flatten(out4, 1))
        
        return proj_output, feature
        '''
        if get_feature:
            return out, feature
        elif get_features:
            features = [
                out0 if not self.return_prerelu else out0_pre,
                out1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out4 if not self.return_prerelu else self.layer4[-1].prerelu,
                ]
            return out, features
        
        else:
            return out
        '''
    
    def _forward_F(self, x):
        out0 = self.layer4[1](x)
        out1 = self.avgpool(out0)
        feature = torch.flatten(out1, 1)
        out = self.fc(feature)
        return out

    def _forward_G(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out0 = self.maxpool(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4[0](out3)
        feature = torch.flatten(out4, 1)
        out = self.fc(feature)
        return out

    def forward(self, x, get_feature=False, get_features=False, F=None, G=None):
        if F:
            return self._forward_F(x) # last 
        elif G:
            return self._forward_G(x)
        else:
            return self._forward_impl(x)

class ResNet_G(ResNet):
    def __init__(self, block, layers):
        super(ResNet,self).__init__(block, layers)
        del self.layer4[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out0 = self.maxpool(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4[0](out3)
        feature = torch.flatten(out4, 1)
        out = self.fc(feature)
        return out
    
class ResNet_F(ResNet):
    def __init__(self, block, layers):
        super(ResNet,self).__init__(block, layers)

    def forward(self, x):
        out0 = self.layer4[1](x)
        out1 = self.avgpool(out0)
        feature = torch.flatten(out1, 1)
        out = self.fc(feature)
        return out
