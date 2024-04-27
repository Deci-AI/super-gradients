import math
from typing import Tuple, Callable

import torch
from torch import nn, Tensor

from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules import ConvBNReLU
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.modules.utils import width_multiplier


@register_detection_module()
class YoloNASRDFLHead(BaseDetectionModule, SupportsReplaceNumClasses):
    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        width_mult: float,
        first_conv_group_size: int,
        num_classes: int,
        stride: int,
        reg_max: int,
        cls_dropout_rate: float = 0.0,
        reg_dropout_rate: float = 0.0,
    ):
        """
        Initialize the YoloNASRDFLHead
        :param in_channels: Input channels
        :param inter_channels: Intermediate number of channels
        :param width_mult: Width multiplier
        :param first_conv_group_size: Group size
        :param num_classes: Number of detection classes
        :param stride: Output stride for this head
        :param reg_max: Number of bins in the regression head
        :param cls_dropout_rate: Dropout rate for the classification head
        :param reg_dropout_rate: Dropout rate for the regression head
        """
        super().__init__(in_channels)

        inter_channels = width_multiplier(inter_channels, width_mult, 8)
        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = inter_channels // first_conv_group_size

        self.num_classes = num_classes
        self.stem = ConvBNReLU(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.cls_convs = nn.Sequential(*first_cls_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_reg_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.reg_convs = nn.Sequential(*first_reg_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.cls_pred = nn.Conv2d(inter_channels, self.num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(inter_channels, 2 * (reg_max + 1), 1, 1, 0)
        self.rot_pred = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.offset_pred = nn.Conv2d(inter_channels, 2, kernel_size=1, stride=1, padding=0)

        self.cls_dropout_rate = nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        self.reg_dropout_rate = nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()

        self.stride = stride

        self.prior_prob = 1e-2
        self._initialize_biases()

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        self.cls_pred = compute_new_weights_fn(self.cls_pred, num_classes)
        self.num_classes = num_classes

    @property
    def out_channels(self):
        return None

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        :return Tuple of 4 tensors:
                - reg_output - [B, 2 * (reg_max + 1), H, W] - Size regression for rotated boxes
                - cls_output - [B, C, H, W] - Class logits
                - offset_output [B, 2, H, W]
                - rot_output [B, 1, H, W]
        """
        x = self.stem(x)

        cls_feat = self.cls_convs(x)
        cls_feat = self.cls_dropout_rate(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(x)
        reg_feat = self.reg_dropout_rate(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        rot_output = self.rot_pred(reg_feat)
        offset_output = self.offset_pred(reg_feat)

        return reg_output, cls_output, offset_output, rot_output

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        # torch.nn.init.zeros_(self.cls_pred.weight)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)

        torch.nn.init.zeros_(self.offset_pred.weight)
        torch.nn.init.zeros_(self.offset_pred.bias)
