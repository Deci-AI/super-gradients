import math
from functools import partial
from typing import Tuple, Callable

import torch
from torch import nn, Tensor

from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules import ConvBNReLU, QARepVGGBlock
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
        bbox_inter_channels: int,
        pose_inter_channels: int,
        pose_regression_blocks: int,
        shared_stem: bool,
        pose_conf_in_class_head: bool,
        pose_block_use_repvgg: bool,
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
        :param bbox_inter_channels: Intermediate number of channels for box detection & regression
        :param pose_inter_channels: Intermediate number of channels for pose regression
        :param shared_stem: Whether to share the stem between the pose and bbox heads
        :param pose_conf_in_class_head: Whether to include the pose confidence in the classification head
        :param width_mult: Width multiplier
        :param first_conv_group_size: Group size
        :param num_classes: Number of keypoints classes for pose regression. Number of detection classes is always 1.
        :param stride: Output stride for this head
        :param reg_max: Number of bins in the regression head
        :param cls_dropout_rate: Dropout rate for the classification head
        :param reg_dropout_rate: Dropout rate for the regression head
        """
        super().__init__(in_channels)

        bbox_inter_channels = width_multiplier(bbox_inter_channels, width_mult, 8)
        pose_inter_channels = width_multiplier(pose_inter_channels, width_mult, 8)

        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = bbox_inter_channels // first_conv_group_size

        self.num_classes = num_classes
        self.shared_stem = shared_stem
        self.pose_conf_in_class_head = pose_conf_in_class_head

        if self.shared_stem:
            max_input = max(bbox_inter_channels, pose_inter_channels)
            self.stem = ConvBNReLU(in_channels, max_input, kernel_size=1, stride=1, padding=0, bias=False)

            if max_input != pose_inter_channels:
                self.pose_stem = nn.Conv2d(max_input, pose_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.pose_stem = nn.Identity()

            if max_input != bbox_inter_channels:
                self.bbox_stem = nn.Conv2d(max_input, bbox_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.bbox_stem = nn.Identity()

        else:
            self.stem = nn.Identity()
            self.pose_stem = ConvBNReLU(in_channels, pose_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bbox_stem = ConvBNReLU(in_channels, bbox_inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        first_cls_conv = [ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.cls_convs = nn.Sequential(*first_cls_conv, ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        first_reg_conv = [ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.reg_convs = nn.Sequential(*first_reg_conv, ConvBNReLU(bbox_inter_channels, bbox_inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        if pose_block_use_repvgg:
            pose_block = partial(QARepVGGBlock, use_alpha=True)
        else:
            pose_block = partial(ConvBNReLU, kernel_size=3, stride=1, padding=1, bias=False)

        pose_convs = [pose_block(pose_inter_channels, pose_inter_channels) for _ in range(pose_regression_blocks)]
        self.pose_convs = nn.Sequential(*pose_convs)

        self.reg_pred = nn.Conv2d(bbox_inter_channels, 4 * (reg_max + 1), 1, 1, 0)

        if self.pose_conf_in_class_head:
            self.cls_pred = nn.Conv2d(bbox_inter_channels, 1 + self.num_classes, 1, 1, 0)
            self.pose_pred = nn.Conv2d(pose_inter_channels, 2 * self.num_classes, 1, 1, 0)  # each keypoint is x,y
        else:
            self.cls_pred = nn.Conv2d(bbox_inter_channels, 1, 1, 1, 0)
            self.pose_pred = nn.Conv2d(pose_inter_channels, 3 * self.num_classes, 1, 1, 0)  # each keypoint is x,y,confidence

        self.cls_dropout_rate = nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        self.reg_dropout_rate = nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()

        self.stride = stride

        self.prior_prob = 1e-2
        self._initialize_biases()

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        if self.pose_conf_in_class_head:
            self.cls_pred = compute_new_weights_fn(self.cls_pred, 1 + num_classes)
            self.pose_pred = compute_new_weights_fn(self.pose_pred, 2 * num_classes)
        else:
            self.pose_pred = compute_new_weights_fn(self.pose_pred, 3 * num_classes)
        self.num_classes = num_classes

    @property
    def out_channels(self):
        return None

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """

        :param x: Input feature map of shape [B, Cin, H, W]
        :return: Tuple of [reg_output, cls_output, pose_regression, pose_logits]
            - reg_output:      Tensor of [B, 4 * (reg_max + 1), H, W]
            - cls_output:      Tensor of [B, 1, H, W]
            - pose_regression: Tensor of [B, num_classes, 2, H, W]
            - pose_logits:     Tensor of [B, num_classes, H, W]
        """
        x = self.stem(x)
        pose_features = self.pose_stem(x)
        bbox_features = self.bbox_stem(x)

        cls_feat = self.cls_convs(bbox_features)
        cls_feat = self.cls_dropout_rate(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(bbox_features)
        reg_feat = self.reg_dropout_rate(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        pose_feat = self.pose_convs(pose_features)
        pose_feat = self.reg_dropout_rate(pose_feat)

        pose_output = self.pose_pred(pose_feat)

        if self.pose_conf_in_class_head:
            pose_logits = cls_output[:, 1:, :, :]
            cls_output = cls_output[:, 0:1, :, :]
            pose_regression = pose_output.reshape((pose_output.size(0), self.num_classes, 2, pose_output.size(2), pose_output.size(3)))
        else:
            pose_output = pose_output.reshape((pose_output.size(0), self.num_classes, 3, pose_output.size(2), pose_output.size(3)))
            pose_logits = pose_output[:, :, 2, :, :]
            pose_regression = pose_output[:, :, 0:2, :, :]

        return reg_output, cls_output, pose_regression, pose_logits

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)
