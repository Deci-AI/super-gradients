"""
This file is used to define the model used for training. For example, in this template, we define ResNet50.
One may use existing models from torchvision as well (e.g., torchvision.models.resnet50)
"""

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def width_multiplier(original, factor):
    return int(original * factor)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks: list, num_classes: int = 10, width_mult: float = 1,
                 input_batchnorm: bool = False, backbone_mode: bool = False):
        super(ResNet, self).__init__()
        self.backbone_mode = backbone_mode
        self.structure = [num_blocks, width_mult]
        self.in_planes = width_multiplier(64, width_mult)
        self.input_batchnorm = input_batchnorm
        if self.input_batchnorm:
            self.bn0 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, width_multiplier(64, width_mult), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(width_multiplier(64, width_mult))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, width_multiplier(64, width_mult), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width_multiplier(128, width_mult), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, width_multiplier(256, width_mult), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, width_multiplier(512, width_mult), num_blocks[3], stride=2)

        if not self.backbone_mode:
            # IF RESNET IS IN BACK_BONE MODE WE DON'T NEED THE FINAL CLASSIFIER LAYERS, BUT ONLY THE NET BLOCK STRUCTURE
            self.linear = nn.Linear(width_multiplier(512, width_mult) * block.expansion, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if num_blocks == 0:
            # When the number of blocks is zero but spatial dimension and/or number of filters about to change we put 1
            # 3X3 conv layer to make this change to the new dimensions.
            if stride != 1 or self.in_planes != planes:
                layers.append(nn.Sequential(
                    nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=stride, bias=False, padding=1),
                    nn.BatchNorm2d(planes))
                )
                self.in_planes = planes

        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.input_batchnorm:
            x = self.bn0(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if not self.backbone_mode:
            # IF RESNET IS *NOT* IN BACK_BONE MODE WE  NEED THE FINAL CLASSIFIER LAYERS OUTPUTS
            out = self.avgpool(out)
            out = out.squeeze(dim=2).squeeze(dim=2)
            out = self.linear(out)

        return out

    def load_state_dict(self, state_dict, strict=True):
        """
        load_state_dict - Overloads the base method and calls it to load a modified dict for usage as a backbone
        :param state_dict:  The state_dict to load
        :param strict:      strict loading (see super() docs)
        """
        pretrained_model_weights_dict = state_dict.copy()

        if self.backbone_mode:
            # FIRST LET'S POP THE LAST TWO LAYERS - NO NEED TO LOAD THEIR VALUES SINCE THEY ARE IRRELEVANT AS A BACKBONE
            pretrained_model_weights_dict.popitem()
            pretrained_model_weights_dict.popitem()

            pretrained_backbone_weights_dict = OrderedDict()
            for layer_name, weights in pretrained_model_weights_dict.items():
                # GET THE LAYER NAME WITHOUT THE 'module.' PREFIX
                name_without_module_prefix = layer_name.split('module.')[1]

                # MAKE SURE THESE ARE NOT THE FINAL LAYERS
                pretrained_backbone_weights_dict[name_without_module_prefix] = weights

            # RETURNING THE UNMODIFIED/MODIFIED STATE DICT DEPENDING ON THE backbone_mode VALUE
            super().load_state_dict(pretrained_backbone_weights_dict, strict)
        else:
            super().load_state_dict(pretrained_model_weights_dict, strict)
