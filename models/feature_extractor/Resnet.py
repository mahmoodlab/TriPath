"""
Resnet 2D model
"""

import torch
import timm
import torch.nn as nn
from collections import OrderedDict

track_running_stats = True

class Basic(nn.Module):
    """
    Basic module used for Resnet18 and Resnet34
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet module for 2D feature extraction that can handle both 2D and 3D patches.
    If input is 3D patch, 2D feature extraction will be performed on each slice of the 3D patch and then averaged

    """
    def __init__(self, block, layers, name='resnet18'):
        self.name = name
        self.channel = 3
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, track_running_stats=track_running_stats),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        threedim = False
        batch_size = x.shape[0]

        if len(x.shape) == 5:  # (B, C, Z, W, H) => 3D
            threedim = True
            x = x.transpose(1, 2)  # (B, C, Z, W, H) => (B, Z, C, W, H)
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.name == 'resnet18' or self.name == 'resnet34':
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if threedim:   # Take average across stacks
            x = x.reshape(batch_size, -1, x.shape[1]) # (B, Z, numOffeatures)
            x = torch.mean(x, dim=1)

        return x

    def get_output_dim(self):
        if self.name == 'resnet18' or self.name == 'resnet34':
            return 512
        else:
            return 1024

    def get_channel_dim(self):
        return self.channel

    def load_weights(self, load_weights=True, pretrained_path=None, **kwargs):
        """
        Load pretrained weights for Resnet2D
        """

        # Load pretrained 2D model if applicable
        if load_weights:
            if 'imagenet' in pretrained_path:
                print("Loading ImageNet weights")
                loaded_model = timm.create_model(self.name, pretrained=True)
                self.load_state_dict(loaded_model.state_dict(), strict=False)
            else:
                od = OrderedDict()
                saved_weights = torch.load(pretrained_path)
                for key, val in saved_weights.items():
                    new_key = '.'.join(key.split('.')[1:])
                    od[new_key] = val

                self.load_state_dict(od, strict=True)


def resnet_2d(encoder='resnet50_2d', trainable_layers=[]):
    """
    Load truncated resnet 2D architecture
    """

    encoder_name = encoder.split('_')[0]

    if encoder_name == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], name=encoder_name)
    elif encoder_name == 'resnet34':
        model = ResNet(Basic, [3, 4, 6, 3], name=encoder_name)
    elif encoder_name == 'resnet18':
        model = ResNet(Basic, [2, 2, 2, 2], name=encoder_name)
    else:
        raise NotImplementedError("{} not implemented!".format(encoder))

    return model


