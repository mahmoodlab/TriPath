"""
Resnet 3D model, which inflates the corresponding 2D version of the model
By default, it disables tracking running stats, since BatchNorm statistics would be quite different from batch statistics
of our 3D dataset
"""

import torch
import timm
import torch.nn as nn
from collections import OrderedDict


class Basic3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=False):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False)
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

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=False):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False)
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

        out = out + residual
        out = self.relu(out)

        return out


class ResnetInflated(nn.Module):
    """
    Inflated version of Resnet
    """
    def __init__(self,
                 block,
                 layers,
                 trainable_layers=[],
                 name='resnet18',
                 track_running_stats=True):
        super().__init__()

        self.name = name
        self.channel = 3
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, track_running_stats=track_running_stats)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, track_running_stats=track_running_stats)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Freeze every parameter
        for param in self.layer1.parameters():
            param.requires_grad = False

        for param in self.layer2.parameters():
            param.requires_grad = False

        for param in self.layer3.parameters():
            param.requires_grad = False

        if len(trainable_layers) > 0:
            self._unfreeze(trainable_layers)

    def _make_layer(self, block, planes, blocks, stride=1, track_running_stats=False):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion, track_running_stats=track_running_stats),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, track_running_stats=track_running_stats))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, track_running_stats=track_running_stats))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.name == 'resnet18' or self.name == 'resnet34':
            out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out

    def get_output_dim(self):
        if self.name == 'resnet18' or self.name == 'resnet34':
            return 512
        else:
            return 1024

    def get_channel_dim(self):
        return self.channel

    def load_weights(self, load_weights=True, pretrained_path=None, **kwargs):
        """
        Load pretrained weights for Resnet3D
        """

        # Load pretrained 2D model if applicable
        if load_weights:
            if 'imagenet' in pretrained_path:
                print("\nLoading ImageNet weights...")
                loaded_model = timm.create_model(self.name, pretrained=True)
                inflate3d(self.state_dict(), loaded_model)
            else:
                od = OrderedDict()
                saved_weights = torch.load(pretrained_path)
                for key, val in saved_weights.items():
                    new_key = '.'.join(key.split('.')[1:])
                    od[new_key] = val

                self.load_state_dict(od, strict=False)

    def _unfreeze(self, trainable_layers=[]):
        """
        Unfreeze parameters in the network
        """

        for layer in trainable_layers:
            if layer == 'all':
                for name, param in self.named_parameters():
                    print("--- {} is now trainable".format(name))
                    param.requires_grad = True
                break

            if layer == 'layer1':
                for name, param in self.layer1.named_parameters():
                    print("--- {} is now trainable".format(name))
                    param.requires_grad = True

            if layer == 'layer2':
                for name, param in self.layer2.named_parameters():
                    print("--- {} is now trainable".format(name))
                    param.requires_grad = True

            if layer == 'layer3':
                for name, param in self.layer3.named_parameters():
                    print("--- {} is now trainable".format(name))
                    param.requires_grad = True


def inflate_layer(model, layer):
    param = model.state_dict()[layer]

    if 'conv' in layer: # Inflate 2D kernel to 3D kernel by copying weights
        z_dim = param.shape[-1] # Equilength kernel
        param = param.unsqueeze(2).repeat(1, 1, z_dim, 1, 1)
        param = param / z_dim
    elif 'downsample.0' in layer: # Inflate 2D kernel to 3D kernel by copying weights
        z_dim = param.shape[-1]  # Equilength kernel
        param = param.unsqueeze(2).repeat(1, 1, z_dim, 1, 1)
        param = param / z_dim

    return param


def inflate3d(model_state_dict, loaded_model):
    """
    Inflate and copy 2D convolutional kernels from pretrained model
    """
    for name, param in model_state_dict.items():
        with torch.no_grad():
            updated_param = inflate_layer(model=loaded_model, layer=name)
            param.copy_(updated_param)


def resnet_3d(encoder='resnet18_3d',
              trainable_layers=[]):
    """
    Main entry point
    Intstantitates Resnet 3D models
    """

    encoder_name = encoder.split('_')[0]

    if encoder_name == 'resnet50':
        model = ResnetInflated(Bottleneck3D,
                               [3, 4, 6, 3],
                               trainable_layers=trainable_layers,
                               name=encoder_name)
    elif encoder_name == 'resnet34':
        model = ResnetInflated(Basic3D,
                               [3, 4, 6, 3],
                               trainable_layers=trainable_layers,
                               name=encoder_name)
    elif encoder_name == 'resnet18':
        model = ResnetInflated(Basic3D,
                               [2, 2, 2, 2],
                               trainable_layers=trainable_layers,
                               name=encoder_name)
    else:
        raise NotImplementedError("{} not implemented!".format(encoder))

    return model



