"""ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.

Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models.sg_module import SgModule


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GroupedConvBlock(nn.Module):
    """Grouped convolution block."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(GroupedConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.norm_layer = norm_layer
        width = int(planes * (base_width / 64.0)) * groups
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
        out = self.relu(out)

        return out


class ResNeXt(SgModule):
    def __init__(self, layers, cardinality, bottleneck_width, num_classes=10, replace_stride_with_dilation=None):
        super(ResNeXt, self).__init__()

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None " "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.cardinality = cardinality
        self.dilation = 1
        self.inplanes = 64
        self.base_width = bottleneck_width
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(GroupedConvBlock, 64, layers[0])
        self.layer2 = self._make_layer(GroupedConvBlock, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(GroupedConvBlock, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if len(layers) == 4:
            self.layer4 = self._make_layer(GroupedConvBlock, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        end_width = 512 if len(layers) == 4 else 256
        self.fc = nn.Linear(end_width * GroupedConvBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
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

        layers = [block(self.inplanes, planes, stride, downsample, self.cardinality, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.cardinality, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.layer4 is not None:
            out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CustomizedResNeXt(ResNeXt):
    def __init__(self, arch_params):
        super(CustomizedResNeXt, self).__init__(
            layers=arch_params.structure if hasattr(arch_params, "structure") else [3, 3, 3],
            bottleneck_width=arch_params.num_init_features if hasattr(arch_params, "bottleneck_width") else 64,
            cardinality=arch_params.bn_size if hasattr(arch_params, "cardinality") else 32,
            num_classes=arch_params.num_classes,
            replace_stride_with_dilation=arch_params.replace_stride_with_dilation if hasattr(arch_params, "replace_stride_with_dilation") else None,
        )


@register_model(Models.RESNEXT50)
class ResNeXt50(ResNeXt):
    def __init__(self, arch_params):
        super(ResNeXt50, self).__init__(layers=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, num_classes=arch_params.num_classes)


@register_model(Models.RESNEXT101)
class ResNeXt101(ResNeXt):
    def __init__(self, arch_params):
        super(ResNeXt101, self).__init__(layers=[3, 4, 23, 3], cardinality=32, bottleneck_width=8, num_classes=arch_params.num_classes)
