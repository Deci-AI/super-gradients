"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Pre-trained ImageNet models: 'deci-model-repository/resnet?/ckpt_best.pth' => ? = the type of resnet (e.g. 18, 34...)
Pre-trained CIFAR10 models: 'deci-model-repository/CIFAR_NAS_#?_????_?/ckpt_best.pth' => ? = num of model, structure, width_mult

Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from super_gradients.modules.utils import width_multiplier
from super_gradients.training.models import SgModule
from super_gradients.training.utils import get_param
from super_gradients.training.utils.regularization_utils import DropPath
from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models


class BasicResNetBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expansion=1, final_relu=True, droppath_prob=0.0):
        super(BasicResNetBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, expansion=4, final_relu=True, droppath_prob=0.0):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.final_relu = final_relu

        self.drop_path = DropPath(drop_prob=droppath_prob)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.drop_path(out)

        out += self.shortcut(x)

        if self.final_relu:
            out = F.relu(out)

        return out


class CifarResNet(SgModule):
    def __init__(self, block, num_blocks, num_classes=10, width_mult=1, expansion=1):
        super(CifarResNet, self).__init__()
        self.expansion = expansion
        self.structure = [num_blocks, width_mult]
        self.in_planes = width_multiplier(64, width_mult)
        self.conv1 = nn.Conv2d(3, width_multiplier(64, width_mult), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width_multiplier(64, width_mult))
        self.layer1 = self._make_layer(block, width_multiplier(64, width_mult), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width_multiplier(128, width_mult), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, width_multiplier(256, width_mult), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, width_multiplier(512, width_mult), num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(width_multiplier(512, width_mult) * self.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if num_blocks == 0:
            # When the number of blocks is zero but spatial dimension and/or number of filters about to change we put 1
            # 3X3 conv layer to make this change to the new dimensions.
            if stride != 1 or self.in_planes != planes:
                layers.append(nn.Sequential(nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=stride, bias=False, padding=1), nn.BatchNorm2d(planes)))
                self.in_planes = planes

        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(SgModule):
    def __init__(
        self,
        block,
        num_blocks: list,
        num_classes: int = 10,
        width_mult: float = 1,
        expansion: int = 1,
        droppath_prob=0.0,
        input_batchnorm: bool = False,
        backbone_mode: bool = False,
    ):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.backbone_mode = backbone_mode
        self.structure = [num_blocks, width_mult]
        self.in_planes = width_multiplier(64, width_mult)
        self.input_batchnorm = input_batchnorm
        if self.input_batchnorm:
            self.bn0 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(3, width_multiplier(64, width_mult), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(width_multiplier(64, width_mult))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, width_multiplier(64, width_mult), num_blocks[0], stride=1, droppath_prob=droppath_prob)
        self.layer2 = self._make_layer(block, width_multiplier(128, width_mult), num_blocks[1], stride=2, droppath_prob=droppath_prob)
        self.layer3 = self._make_layer(block, width_multiplier(256, width_mult), num_blocks[2], stride=2, droppath_prob=droppath_prob)
        self.layer4 = self._make_layer(block, width_multiplier(512, width_mult), num_blocks[3], stride=2, droppath_prob=droppath_prob)

        if not self.backbone_mode:
            # IF RESNET IS IN BACK_BONE MODE WE DON'T NEED THE FINAL CLASSIFIER LAYERS, BUT ONLY THE NET BLOCK STRUCTURE
            self.linear = nn.Linear(width_multiplier(512, width_mult) * self.expansion, num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.width_mult = width_mult

    def _make_layer(self, block, planes, num_blocks, stride, droppath_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if num_blocks == 0:
            # When the number of blocks is zero but spatial dimension and/or number of filters about to change we put 1
            # 3X3 conv layer to make this change to the new dimensions.
            if stride != 1 or self.in_planes != planes:
                layers.append(nn.Sequential(nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=stride, bias=False, padding=1), nn.BatchNorm2d(planes)))
                self.in_planes = planes

        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride, droppath_prob=droppath_prob))
                self.in_planes = planes * self.expansion
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
                name_without_module_prefix = layer_name.split("module.")[1]

                # MAKE SURE THESE ARE NOT THE FINAL LAYERS
                pretrained_backbone_weights_dict[name_without_module_prefix] = weights

            # RETURNING THE UNMODIFIED/MODIFIED STATE DICT DEPENDING ON THE backbone_mode VALUE
            super().load_state_dict(pretrained_backbone_weights_dict, strict)
        else:
            super().load_state_dict(pretrained_model_weights_dict, strict)

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.linear = new_head
        else:
            self.linear = nn.Linear(width_multiplier(512, self.width_mult) * self.expansion, new_num_classes)


@register_model(Models.RESNET18)
class ResNet18(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            BasicResNetBlock,
            [2, 2, 2, 2],
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
        )


@register_model(Models.RESNET18_CIFAR)
class ResNet18Cifar(CifarResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(BasicResNetBlock, [2, 2, 2, 2], num_classes=num_classes or arch_params.num_classes)


@register_model(Models.RESNET34)
class ResNet34(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            BasicResNetBlock,
            [3, 4, 6, 3],
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
        )


@register_model(Models.RESNET50)
class ResNet50(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            Bottleneck,
            [3, 4, 6, 3],
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
            expansion=4,
        )


@register_model(Models.RESNET50_3343)
class ResNet50_3343(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            Bottleneck,
            [3, 3, 4, 3],
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
            expansion=4,
        )


@register_model(Models.RESNET101)
class ResNet101(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
            expansion=4,
        )


@register_model(Models.RESNET152)
class ResNet152(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            Bottleneck,
            [3, 8, 36, 3],
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
            expansion=4,
        )


@register_model(Models.CUSTOM_RESNET_CIFAR)
class CustomizedResnetCifar(CifarResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(BasicResNetBlock, arch_params.structure, width_mult=arch_params.width_mult, num_classes=num_classes or arch_params.num_classes)


@register_model(Models.CUSTOM_RESNET50_CIFAR)
class CustomizedResnet50Cifar(CifarResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(Bottleneck, arch_params.structure, width_mult=arch_params.width_mult, num_classes=num_classes or arch_params.num_classes, expansion=4)


@register_model(Models.CUSTOM_RESNET)
class CustomizedResnet(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            BasicResNetBlock,
            arch_params.structure,
            width_mult=arch_params.width_mult,
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
        )


@register_model(Models.CUSTOM_RESNET50)
class CustomizedResnet50(ResNet):
    def __init__(self, arch_params, num_classes=None):
        super().__init__(
            Bottleneck,
            arch_params.structure,
            width_mult=arch_params.width_mult,
            num_classes=num_classes or arch_params.num_classes,
            droppath_prob=get_param(arch_params, "droppath_prob", 0),
            backbone_mode=get_param(arch_params, "backbone_mode", False),
            expansion=4,
        )
