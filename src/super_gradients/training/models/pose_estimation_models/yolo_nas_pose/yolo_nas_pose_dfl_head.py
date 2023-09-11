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
class YoloNASPoseDFLHead(BaseDetectionModule, SupportsReplaceNumClasses):
    """
    YoloNASPoseDFLHead is the head used in YoloNASPose model.
    This class implements single-class object detection and keypoints regression on a single scale feature map
    """

    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        pose_inter_channels: int,
        width_mult: float,
        first_conv_group_size: int,
        num_classes: int,
        stride: int,
        reg_max: int,
        cls_dropout_rate: float = 0.0,
        reg_dropout_rate: float = 0.0,
    ):
        """
        Initialize the YoloNASDFLHead
        :param in_channels: Input channels
        :param inter_channels: Intermediate number of channels
        :param width_mult: Width multiplier
        :param first_conv_group_size: Group size
        :param num_classes: Number of keypoints classes for pose regression. Number of detection classes is always 1.
        :param stride: Output stride for this head
        :param reg_max: Number of bins in the regression head
        :param cls_dropout_rate: Dropout rate for the classification head
        :param reg_dropout_rate: Dropout rate for the regression head
        """
        super().__init__(in_channels)

        inter_channels = width_multiplier(inter_channels, width_mult, 8)
        pose_inter_channels = width_multiplier(pose_inter_channels, width_mult, 8)
        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = inter_channels // first_conv_group_size

        self.stride = stride
        self.reg_max = reg_max
        self.num_classes = num_classes

        self.stem = ConvBNReLU(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.cls_convs = nn.Sequential(*first_cls_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_reg_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.reg_convs = nn.Sequential(*first_reg_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_pose_conv = [ConvBNReLU(in_channels, pose_inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.pose_convs = nn.Sequential(
            *first_pose_conv,
            ConvBNReLU(pose_inter_channels, pose_inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            ConvBNReLU(pose_inter_channels, pose_inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.cls_pred = nn.Conv2d(inter_channels, 1 + self.num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(inter_channels, 4 * (reg_max + 1), 1, 1, 0)
        self.pose_pred = nn.Conv2d(pose_inter_channels, 2 * self.num_classes * (reg_max + 1), 1, 1, 0)  # each keypoint is x,y,confidence

        self.cls_dropout_rate = nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        self.reg_dropout_rate = nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()

        self.prior_prob = 1e-2
        self._initialize_biases()

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        self.cls_pred = compute_new_weights_fn(self.cls_pred, 1 + num_classes)
        self.pose_pred = compute_new_weights_fn(self.pose_pred, 2 * self.num_classes * (self.reg_max + 1))
        self.num_classes = num_classes

    @property
    def out_channels(self):
        return None

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        :param x: Input feature map of shape [B, Cin, H, W]
        :return: Tuple of [reg_output, cls_output, pose_output]
            - reg_output: Tensor of [B, 4 * (reg_max + 1), H, W]
            - cls_output: Tensor of [B, 1, H, W]
            - pose_logits: Tensor of [B, num_classes, H, W]
            - pose_output: Tensor of [B, num_bins, num_classes, 2, H, W]
        """
        box_feat = self.stem(x)

        cls_feat = self.cls_convs(box_feat)
        cls_feat = self.cls_dropout_rate(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        pose_logits = cls_output[:, 1:, :, :]
        cls_output = cls_output[:, 0:1, :, :]

        reg_feat = self.reg_convs(box_feat)
        reg_feat = self.reg_dropout_rate(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        pose_feat = self.pose_convs(x)
        pose_feat = self.reg_dropout_rate(pose_feat)
        pose_output = self.pose_pred(pose_feat)

        pose_output = pose_output.reshape((pose_output.size(0), self.reg_max + 1, self.num_classes, 2, pose_output.size(2), pose_output.size(3)))
        return reg_output, cls_output, pose_logits, pose_output

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)
