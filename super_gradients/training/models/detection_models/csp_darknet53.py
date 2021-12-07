"""
CSP Darknet

credits: https://github.com/ultralytics
"""
import math
from typing import Tuple, Type

import torch
import torch.nn as nn

from super_gradients.training.utils.utils import get_param, HpmStruct
from super_gradients.training.models.sg_module import SgModule


def autopad(kernel, padding=None):
    # PAD TO 'SAME'
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding


def width_multiplier(original, factor, divisor: int = None):
    if divisor is None:
        return int(original * factor)
    else:
        return math.ceil(int(original * factor) / divisor) * divisor


def get_yolo_version_params(yolo_version: str, width_mult_factor: float, depth_mult_factor: float):
    if yolo_version == 'v6.0':
        struct = (3, 6, 9, 3)
        block = C3
        activation_type = nn.SiLU
        width_mult = lambda channels: width_multiplier(channels, width_mult_factor, 8)
    elif yolo_version == 'v3.0':
        struct = (3, 9, 9, 3)
        block = BottleneckCSP
        activation_type = nn.Hardswish
        width_mult = lambda channels: width_multiplier(channels, width_mult_factor)
    else:
        raise NotImplementedError(f'YoloV5 release version {yolo_version} is not supported, use "v3.0" or "v6.0"')

    depth_mult = lambda blocks: max(round(blocks * depth_mult_factor), 1) if blocks > 1 else blocks
    return struct, block, activation_type, width_mult, depth_mult


class NumClassesMissingException(Exception):
    pass


class Conv(nn.Module):
    # STANDARD CONVOLUTION
    def __init__(self, input_channels, output_channels, kernel, stride, activation_type: Type[nn.Module],
                 padding: int = None, groups: int = 1):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel, stride, autopad(kernel, padding), groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = activation_type()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # STANDARD BOTTLENECK
    def __init__(self, input_channels, output_channels, shortcut: bool, activation_type: Type[nn.Module], groups=1):
        super().__init__()

        hidden_channels = output_channels
        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(hidden_channels, output_channels, 3, 1, activation_type, groups=groups)
        self.add = shortcut and input_channels == output_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions https://github.com/ultralytics/yolov5
    def __init__(self, input_channels, output_channels, bottleneck_blocks_num, activation_type: Type[nn.Module],
                 shortcut=True, groups=1, expansion=0.5):
        super().__init__()

        hidden_channels = int(output_channels * expansion)

        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv3 = Conv(2 * hidden_channels, output_channels, 1, 1, activation_type)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, activation_type, groups=groups)
                                 for _ in range(bottleneck_blocks_num)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, input_channels, output_channels, bottleneck_blocks_num, activation_type: Type[nn.Module],
                 shortcut=True, groups=1, expansion=0.5):
        super().__init__()

        hidden_channels = int(output_channels * expansion)

        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = nn.Conv2d(input_channels, hidden_channels, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.cv4 = Conv(2 * hidden_channels, output_channels, 1, 1, activation_type)
        self.bn = nn.BatchNorm2d(2 * hidden_channels)  # APPLIED TO CAT(CV2, CV3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, activation_type, groups=groups)
                                 for _ in range(bottleneck_blocks_num)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # SPATIAL PYRAMID POOLING LAYER USED IN YOLOV3-SPP
    def __init__(self, input_channels, output_channels, k: Tuple, activation_type: Type[nn.Module]):
        super().__init__()

        hidden_channels = input_channels // 2
        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(hidden_channels * (len(k) + 1), output_channels, 1, 1, activation_type)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher https://github.com/ultralytics/yolov5
    # equivalent to SPP(k=(5, 9, 13))
    def __init__(self, input_channels, output_channels, k: int, activation_type: Type[nn.Module]):
        super().__init__()

        hidden_channels = input_channels // 2  # hidden channels
        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(hidden_channels * 4, output_channels, 1, 1, activation_type)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.cv2(torch.cat([x, y1, y2, self.maxpool(y2)], 1))


class Focus(nn.Module):
    # FOCUS WH INFORMATION INTO C-SPACE
    def __init__(self, input_channels, output_channels, kernel, stride, activation_type: Type[nn.Module],
                 padding=None, groups=1):
        super().__init__()

        self.conv = Conv(input_channels * 4, output_channels, kernel, stride, activation_type, padding, groups)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class ViewModule(nn.Module):
    """
    Returns a reshaped version of the input, to be used in None-Backbone Mode
    """

    def __init__(self, features=1024):
        super(ViewModule, self).__init__()
        self.features = features

    def forward(self, x):
        return x.view(-1, self.features)


class CSPDarknet53(SgModule):
    def __init__(self, arch_params: HpmStruct):
        super().__init__()
        self.num_classes = arch_params.num_classes
        self.backbone_mode = get_param(arch_params, 'backbone_mode', False)
        depth_mult_factor = get_param(arch_params, 'depth_mult_factor', 1.)
        width_mult_factor = get_param(arch_params, 'width_mult_factor', 1.)
        channels_in = get_param(arch_params, 'channels_in', 3)
        yolo_version = get_param(arch_params, 'yolo_version', 'v6.0')

        struct, block, activation_type, width_mult, depth_mult = get_yolo_version_params(yolo_version,
                                                                                         width_mult_factor,
                                                                                         depth_mult_factor)

        struct = [depth_mult(s) for s in struct]
        self._modules_list = nn.ModuleList()

        if yolo_version == 'v6.0':
            self._modules_list.append(Conv(channels_in, width_mult(64), 6, 2, activation_type, padding=2))    # 0
        elif yolo_version == 'v3.0':
            self._modules_list.append(Focus(channels_in, width_mult(64), 3, 1, activation_type))              # 0
        else:
            raise NotImplementedError(f'YoloV5 release version {yolo_version} is not supported, use "v3.0" or "v6.0"')

        self._modules_list.append(Conv(width_mult(64), width_mult(128), 3, 2, activation_type))                      # 1
        self._modules_list.append(block(width_mult(128), width_mult(128), struct[0], activation_type))               # 2
        self._modules_list.append(Conv(width_mult(128), width_mult(256), 3, 2, activation_type))                     # 3
        self._modules_list.append(block(width_mult(256), width_mult(256), struct[1], activation_type))               # 4
        self._modules_list.append(Conv(width_mult(256), width_mult(512), 3, 2, activation_type))                     # 5
        self._modules_list.append(block(width_mult(512), width_mult(512), struct[2], activation_type))               # 6
        self._modules_list.append(Conv(width_mult(512), width_mult(1024), 3, 2, activation_type))                    # 7
        if yolo_version == 'v6.0':
            self._modules_list.append(block(width_mult(1024), width_mult(1024), struct[3], activation_type))         # 8
            self._modules_list.append(SPPF(width_mult(1024), width_mult(1024), 5, activation_type))                  # 9
        elif yolo_version == 'v3.0':
            self._modules_list.append(SPP(width_mult(1024), width_mult(1024), (5, 9, 13), activation_type))          # 8
            self._modules_list.append(block(width_mult(1024), width_mult(1024), struct[3], activation_type, False))  # 9
        else:
            raise NotImplementedError(f'YoloV5 release version {yolo_version} is not supported, use "v3.0" or "v6.0"')

        if not self.backbone_mode:
            # IF NOT USED AS A BACKEND BUT AS A CLASSIFIER WE ADD THE CLASSIFICATION LAYERS
            self._modules_list.append(nn.AdaptiveAvgPool2d((1, 1)))
            self._modules_list.append(ViewModule(1024))
            self._modules_list.append(nn.Linear(1024, self.num_classes))

    def forward(self, x):
        return self._modules_list(x)
