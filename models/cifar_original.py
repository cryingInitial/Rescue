import torch.nn as nn
from models.layers import ConvBlock, InitialBlock, FinalBlock
from collections import OrderedDict
# from mmcv.cnn import build_norm_layer

# class MLPFFNNeck(nn.Module):
#     def __init__(self, in_channels=512, out_channels=512):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d((1, 1))
#         self.ln1 = nn.Sequential(OrderedDict([
#             ('linear', nn.Linear(in_channels, in_channels * 2)),
#             ('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
#             ('relu', nn.LeakyReLU(0.1))
#         ]))
#         self.ln2 = nn.Sequential(OrderedDict([
#             ('linear', nn.Linear(in_channels * 2, in_channels * 2)),
#             ('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
#             ('relu', nn.LeakyReLU(0.1))
#         ]))
#         self.ln3 = nn.Sequential(OrderedDict([
#             ('linear', nn.Linear(in_channels * 2, out_channels, bias=False)),
#         ]))
#         if in_channels == out_channels:
#             # self.ffn = nn.Identity()
#             self.ffn = nn.Sequential(OrderedDict([
#                 ('proj', nn.Linear(in_channels, out_channels, bias=False)),
#             ]))
#         else:
#             self.ffn = nn.Sequential(OrderedDict([
#                 ('proj', nn.Linear(in_channels, out_channels, bias=False)),
#             ]))

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        expansion = 1
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            last=True
        )
        self.activate = getattr(nn, opt.activetype)()

    def forward(self, input_list):
        x, features, get_features, detached = input_list
        _out = self.conv1(x)
        _out = self.conv2(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x

        _out = self.activate(_out) + shortcut

        if get_features:
            d_out = self.conv1(x.detach())
            d_out = self.conv2(d_out)
            if self.downsample is not None:
                d_shortcut = self.downsample(x.detach())
            else:
                d_shortcut = x.detach()
            d_out = d_out + d_shortcut
            features.append(d_out)
        return [_out, features, get_features, detached]


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        expansion = 4
        self.conv1 = ConvBlock(
            opt=opt,
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv3 = ConvBlock(
            opt=opt,
            in_channels=outChannels,
            out_channels=outChannels * expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.downsample = downsample

    def forward(self, input_list):
        x, features, get_features, detached = input_list
        _out = self.conv1(x)
        _out = self.conv2(_out)
        _out = self.conv3(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        _out = _out + shortcut
        if get_features:
            if detached:
                d_out = self.conv1(x.detach())
                d_out = self.conv2(d_out)
                d_out = self.conv3(d_out)
                if self.downsample is not None:
                    d_shortcut = self.downsample(x.detach())
                else:
                    d_shortcut = x.detach()
                d_out = d_out + d_shortcut
                features.append(d_out)
            else:
                features.append(_out)
        return [_out, features, get_features, detached]


class ResidualBlock(nn.Module):
    def __init__(self, opt, block, inChannels, outChannels, depth, stride=1):
        super(ResidualBlock, self).__init__()
        if stride != 1 or inChannels != outChannels * block.expansion:
            downsample = ConvBlock(
                opt=opt,
                in_channels=inChannels,
                out_channels=outChannels * block.expansion,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            downsample = None
        self.blocks = nn.Sequential()
        self.blocks.add_module(
            "block0", block(opt, inChannels, outChannels, stride, downsample)
        )
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module(
                "block{}".format(i), block(opt, inChannels, outChannels)
            )

    def forward(self, x, features=None, get_features=False, detached=False):
        return self.blocks([x, features, get_features, detached])[:2]


class ResNet(nn.Module):
    def __init__(self, opt, model_imagenet=False):
        super(ResNet, self).__init__()
        depth = opt.depth
        self.model_imagenet = model_imagenet
        if depth in [20, 32, 44, 56, 110, 1202]:
            blocktype, self.nettype = "BasicBlock", "cifar"
        elif depth in [164, 1001]:
            blocktype, self.nettype = "BottleneckBlock", "cifar"
        elif depth in [18, 34]:
            blocktype, self.nettype = "BasicBlock", "imagenet"
        elif depth in [50, 101, 152]:
            blocktype, self.nettype = "BottleneckBlock", "imagenet"
        assert depth in [20, 32, 44, 56, 110, 1202, 164, 1001, 18, 34, 50, 101, 152]

        if blocktype == "BasicBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 6 == 0, (
                "Depth should be 6n+2, and preferably one of 20, 32, 44, 56, 110, 1202"
            )
            n = (depth - 2) // 6
            block = BasicBlock
            in_planes, out_planes = 16, 64
        elif blocktype == "BottleneckBlock" and self.nettype == "cifar":
            assert (
                depth - 2
            ) % 9 == 0, "Depth should be 9n+2, and preferably one of 164 or 1001"
            n = (depth - 2) // 9
            block = BottleneckBlock
            in_planes, out_planes = 16, 64
        elif blocktype == "BasicBlock" and self.nettype == "imagenet":
            assert depth in [18, 34]
            num_blocks = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
            block = BasicBlock
            in_planes, out_planes = 64, 512  # 20, 160
        elif blocktype == "BottleneckBlock" and self.nettype == "imagenet":
            assert depth in [50, 101, 152]
            if depth == 50:
                num_blocks = [3, 4, 6, 3]
            elif depth == 101:
                num_blocks = [3, 4, 23, 3]
            elif depth == 152:
                num_blocks = [3, 8, 36, 3]
            block = BottleneckBlock
            in_planes, out_planes = 64, 512
        else:
            assert 1 == 2

        self.num_classes = opt.num_classes
        if model_imagenet:
            self.initial = InitialBlock(
                opt=opt, out_channels=in_planes, kernel_size=7, stride=2, padding=3
            )            
            self.maxpool = nn.MaxPool2d(kernel_size=3,  stride=2, padding=1)
        else:
            self.initial = InitialBlock(
                opt=opt, out_channels=in_planes, kernel_size=3, stride=1, padding=1
            )
        if self.nettype == "cifar":
            self.group1 = ResidualBlock(opt, block, 16, 16, n, stride=1)
            self.group2 = ResidualBlock(
                opt, block, 16 * block.expansion, 32, n, stride=2
            )
            self.group3 = ResidualBlock(
                opt, block, 32 * block.expansion, 64, n, stride=2
            )
        elif self.nettype == "imagenet":
            self.group1 = ResidualBlock(
                opt, block, 64, 64, num_blocks[0], stride=1
            )  # For ResNet-S, convert this to 20,20
            self.group2 = ResidualBlock(
                opt, block, 64 * block.expansion, 128, num_blocks[1], stride=2
            )  # For ResNet-S, convert this to 20,40
            self.group3 = ResidualBlock(
                opt, block, 128 * block.expansion, 256, num_blocks[2], stride=2
            )  # For ResNet-S, convert this to 40,80
            self.group4 = ResidualBlock(
                opt, block, 256 * block.expansion, 512, num_blocks[3], stride=2
            )  # For ResNet-S, convert this to 80,160
        else:
            assert 1 == 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dim_out = out_planes * block.expansion
        #self.fc = FinalBlock(opt=opt, in_channels=out_planes * block.expansion)
        self.fc = nn.Linear(out_planes * block.expansion, self.num_classes)

        # neck 
        # self.neck = MLPFFNNeck(in_channels = 512, out_channels=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, get_feature=False, get_features=False, detached=False):

        last_layer_features = []
        features = []
        out_init = self.initial(x)
        if self.model_imagenet:
            out_init = self.maxpool(out_init)
        
        if get_features:
            features.append(out_init)
            #print("len0", len(last_layer_features))
            last_layer_features.append(features[-1].detach())

        out1, features = self.group1(out_init, features, get_features, detached)

        if get_features:
            #print("len1", len(last_layer_features))
            last_layer_features.append(features[-1].detach())

        out2, features = self.group2(out1, features, get_features, detached)

        if get_features:
            #print("len2", len(last_layer_features))
            last_layer_features.append(features[-1].detach())

        out3, features = self.group3(out2, features, get_features, detached)

        if get_features:
            #print("len3", len(last_layer_features))
            last_layer_features.append(features[-1].detach())

        if self.nettype == "imagenet":
            out4, features = self.group4(out3, features, get_features, detached)
            if get_features:
                #print("len4", len(last_layer_features))
                last_layer_features.append(features[-1].detach())

            feature = self.pool(out4)
        else:
            feature = self.pool(out3)
        
        # neck layer insert (for neural collapse)
        # feature = self.neck(feature)

        feature = feature.view(x.size(0), -1)
        out = self.fc(feature)
        if get_feature:
            return out, feature
        elif get_features:
            return out, last_layer_features
        else:
            return out
