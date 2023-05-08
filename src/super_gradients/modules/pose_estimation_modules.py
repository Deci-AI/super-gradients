from typing import Type, Tuple, Union, List

import torch
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.activations_type_factory import ActivationsTypeFactory
from torch import nn, Tensor

from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.common.registry.registry import register_detection_module


@register_detection_module()
class LightweightDEKRHead(BaseDetectionModule):
    """
    Prediction head for pose estimation task that mimics approach from DEKR (https://arxiv.org/abs/2104.02300) paper,
    but does not use deformable convolutions.

    This head takes single feature map of [B,C,H,W] shape as input and outputs a tuple of (heatmap, offset):
      - heatmap (B, NumJoints+1,H * upsample_factor, W * upsample_factor)
      - offset (B, NumJoints*2, H * upsample_factor, W * upsample_factor)
    """

    @resolve_param("activation", ActivationsTypeFactory())
    def __init__(
        self,
        in_channels: List[int],
        feature_map_index: int,
        num_classes: int,
        heatmap_channels: int,
        offset_channels_per_joint: int,
        activation: Type[nn.Module],
        upscale_factor: int = 1,
    ):
        """

        :param in_channels: Number of input channels.
        :param num_classes: Number of joints to regress.
        :param heatmap_channels: Number of embedding dim for heatmap branch.
        :param offset_channels_per_joint: Number of embedding dim for offset branch per each keypoint. Reasonable value is around 8 - 16.
        :param activation: Activation type used in both branches.
        :param upscale_factor: Upsample factor for produced feature maps. This is useful when you have feature map of stride 8 as input
                               and want to produce output feature maps of stride 4. Upsampling happens before last 1x1 convolution.

        """
        super().__init__(in_channels)

        in_channels = in_channels[feature_map_index]
        self.keypoint_channels = offset_channels_per_joint
        self.num_joints = num_classes
        self.feature_map_index = feature_map_index

        self.heatmap = nn.Sequential(
            nn.Conv2d(in_channels, heatmap_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_channels),
            activation(inplace=True),
            nn.Conv2d(heatmap_channels, heatmap_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_channels),
            activation(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=upscale_factor) if upscale_factor > 1 else nn.Identity(),
            nn.Conv2d(heatmap_channels, self.num_joints + 1, kernel_size=1, padding=0),
        )

        self.transition_offset = nn.Sequential(
            nn.Conv2d(in_channels, self.num_joints * offset_channels_per_joint, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.num_joints * offset_channels_per_joint),
            activation(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=upscale_factor) if upscale_factor > 1 else nn.Identity(),
        )

        self.offset_heads = nn.ModuleList([nn.Conv2d(offset_channels_per_joint, 2, kernel_size=1, padding=0) for _ in range(self.num_joints)])
        self.scale_factor = upscale_factor

    def forward(self, x: List[Tensor]) -> Tuple[Tensor, Tensor]:
        x = x[self.feature_map_index]

        heatmap = self.heatmap(x)

        offset_feature = self.transition_offset(x)

        final_offset = []
        for j in range(self.num_joints):
            final_offset.append(self.offset_heads[j](offset_feature[:, j * self.keypoint_channels : (j + 1) * self.keypoint_channels]))
        offset = torch.cat(final_offset, dim=1)

        return heatmap, offset

    @property
    def out_channels(self) -> Union[List[int], int]:
        return None
