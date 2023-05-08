"""
CSP Darknet

"""
from typing import Tuple, Type

import torch
import torch.nn as nn

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.activations_type_factory import ActivationsTypeFactory
from super_gradients.common.object_names import Models
from super_gradients.common.registry.registry import register_model, register_detection_module
from super_gradients.modules import Residual, Conv, BaseDetectionModule
from super_gradients.modules.utils import width_multiplier
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils.utils import get_param, HpmStruct


def get_yolo_type_params(yolo_type: str, width_mult_factor: float, depth_mult_factor: float):
    if yolo_type == "yoloX":
        struct = (3, 9, 9, 3)
        block = CSPLayer
        activation_type = nn.SiLU
        width_mult = lambda channels: width_multiplier(channels, width_mult_factor)
    else:
        raise NotImplementedError(f"Yolo yolo_type {yolo_type} is not supported")

    depth_mult = lambda blocks: max(round(blocks * depth_mult_factor), 1) if blocks > 1 else blocks
    return struct, block, activation_type, width_mult, depth_mult


class NumClassesMissingException(Exception):
    pass


class GroupedConvBlock(nn.Module):
    """
    Grouped Conv KxK -> usual Conv 1x1
    """

    def __init__(self, input_channels, output_channels, kernel, stride, activation_type: Type[nn.Module], padding: int = None, groups: int = None):
        """
        :param groups:  num of groups in the first conv; if None depthwise separable conv will be used
                        (groups = input channels)
        """
        super().__init__()

        self.dconv = Conv(input_channels, input_channels, kernel, stride, activation_type, padding, groups=groups or input_channels)
        self.conv = Conv(input_channels, output_channels, 1, 1, activation_type)

    def forward(self, x):
        return self.conv(self.dconv(x))


class Bottleneck(nn.Module):
    # STANDARD BOTTLENECK
    def __init__(self, input_channels, output_channels, shortcut: bool, activation_type: Type[nn.Module], depthwise=False):
        super().__init__()

        ConvBlock = GroupedConvBlock if depthwise else Conv
        hidden_channels = output_channels
        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = ConvBlock(hidden_channels, output_channels, 3, 1, activation_type)
        self.add = shortcut and input_channels == output_channels
        self.shortcut = Residual() if self.add else None

    def forward(self, x):
        return self.shortcut(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CSPLayer(nn.Module):
    """
    CSP Bottleneck with 3 convolutions

    :param in_channels: int, input channels.
    :param out_channels: int, output channels.
    :param num_bottlenecks: int, number of bottleneck conv layers.
    :param act: Type[nn.module], activation type.
    :param shortcut: bool, whether to apply shortcut (i.e add input to result) in bottlenecks (default=True).
    :param depthwise: bool, whether to use GroupedConvBlock in last conv in bottlenecks (default=False).
    :param expansion: float, determines the number of hidden channels (default=0.5).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int,
        act: Type[nn.Module],
        shortcut: bool = True,
        depthwise: bool = False,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=act)
        self.conv2 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=act)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1, stride=1, activation_type=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, act, depthwise) for _ in range(num_bottlenecks)]
        self.bottlenecks = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, input_channels, output_channels, bottleneck_blocks_num, activation_type: Type[nn.Module], shortcut=True, depthwise=False, expansion=0.5):
        super().__init__()

        hidden_channels = int(output_channels * expansion)

        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = nn.Conv2d(input_channels, hidden_channels, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.cv4 = Conv(2 * hidden_channels, output_channels, 1, 1, activation_type)
        self.bn = nn.BatchNorm2d(2 * hidden_channels)  # APPLIED TO CAT(CV2, CV3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels, shortcut, activation_type, depthwise) for _ in range(bottleneck_blocks_num)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@register_detection_module()
class SPP(BaseDetectionModule):
    # SPATIAL PYRAMID POOLING LAYER
    @resolve_param("activation_type", ActivationsTypeFactory())
    def __init__(self, in_channels, output_channels, k: Tuple, activation_type: Type[nn.Module]):
        super().__init__(in_channels)
        self._output_channels = output_channels

        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(hidden_channels * (len(k) + 1), output_channels, 1, 1, activation_type)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

    @property
    def out_channels(self):
        """
        :return: channels of tensor(s) that will be returned by a module  in forward
        """
        return self._output_channels


class ViewModule(nn.Module):
    """
    Returns a reshaped version of the input, to be used in None-Backbone Mode
    """

    def __init__(self, features=1024):
        super(ViewModule, self).__init__()
        self.features = features

    def forward(self, x):
        return x.view(-1, self.features)


@register_model(Models.CSP_DARKNET53)
class CSPDarknet53(SgModule):
    def __init__(self, arch_params: HpmStruct):
        super().__init__()
        self.num_classes = arch_params.num_classes
        self.backbone_mode = get_param(arch_params, "backbone_mode", False)
        depth_mult_factor = get_param(arch_params, "depth_mult_factor", 1.0)
        width_mult_factor = get_param(arch_params, "width_mult_factor", 1.0)
        channels_in = get_param(arch_params, "channels_in", 3)
        yolo_type = get_param(arch_params, "yolo_type", "yoloX")
        depthwise = get_param(arch_params, "depthwise", False)

        struct, block, activation_type, width_mult, depth_mult = get_yolo_type_params(yolo_type, width_mult_factor, depth_mult_factor)
        ConvBlock = Conv if not depthwise else GroupedConvBlock

        struct = [depth_mult(s) for s in struct]
        self._modules_list = nn.ModuleList()

        if get_param(arch_params, "stem_type") == "6x6" or yolo_type == "yoloX":
            self._modules_list.append(Conv(channels_in, width_mult(64), 6, 2, activation_type, padding=2))  # 0
        else:
            raise NotImplementedError(f"Yolo type: {yolo_type} is not supported")

        for i, layer_in_ch in enumerate([64, 128, 256, 512]):
            self._modules_list.append(ConvBlock(width_mult(layer_in_ch), width_mult(layer_in_ch * 2), 3, 2, activation_type))  # 1,3,5,7
            if i < 3:
                self._modules_list.append(
                    block(width_mult(layer_in_ch * 2), width_mult(layer_in_ch * 2), struct[i], activation_type, depthwise=depthwise)
                )  # 2,4,6

        if yolo_type == "yoloX":
            self._modules_list.append(SPP(width_mult(1024), width_mult(1024), (5, 9, 13), activation_type))  # 8
            self._modules_list.append(block(width_mult(1024), width_mult(1024), struct[3], activation_type, False, depthwise=depthwise))  # 9
        else:
            raise NotImplementedError(f"Yolo type: {yolo_type} is not supported")

        if not self.backbone_mode:
            # IF NOT USED AS A BACKEND BUT AS A CLASSIFIER WE ADD THE CLASSIFICATION LAYERS
            self._modules_list.append(nn.AdaptiveAvgPool2d((1, 1)))
            self._modules_list.append(ViewModule(1024))
            self._modules_list.append(nn.Linear(1024, self.num_classes))

    def forward(self, x):
        return self._modules_list(x)
