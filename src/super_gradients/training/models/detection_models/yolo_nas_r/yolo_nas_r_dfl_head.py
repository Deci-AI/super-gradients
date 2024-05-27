from typing import Tuple

import torch
from super_gradients.common.registry import register_detection_module
from super_gradients.modules.utils import width_multiplier
from super_gradients.training.models.detection_models.yolo_nas import YoloNASDFLHead
from torch import nn, Tensor


@register_detection_module()
class YoloNASRDFLHead(YoloNASDFLHead):
    """
    YoloNASRDFLHead is a YoloNASDFLHead with additional outputs for rotated bounding boxes.
    """

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
        super().__init__(
            in_channels=in_channels,
            inter_channels=inter_channels,
            width_mult=width_mult,
            first_conv_group_size=first_conv_group_size,
            num_classes=num_classes,
            stride=stride,
            reg_max=reg_max,
            cls_dropout_rate=cls_dropout_rate,
            reg_dropout_rate=reg_dropout_rate,
        )
        inter_channels = width_multiplier(inter_channels, width_mult, 8)

        self.reg_pred = nn.Conv2d(inter_channels, 2 * (reg_max + 1), 1, 1, 0)
        self.rot_pred = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.offset_pred = nn.Conv2d(inter_channels, 2, kernel_size=1, stride=1, padding=0)
        torch.nn.init.zeros_(self.offset_pred.weight)
        torch.nn.init.zeros_(self.offset_pred.bias)

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
