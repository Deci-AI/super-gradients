from typing import Tuple, Type, List

import numpy as np
import torch
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.activations_type_factory import ActivationsTypeFactory
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from torch import nn, Tensor

from super_gradients.modules import ConvBNAct
from super_gradients.training.utils.version_utils import torch_version_is_greater_or_equal


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


@torch.no_grad()
def generate_anchors_for_grid_cell(
    feats: Tuple[Tensor, ...],
    fpn_strides: Tuple[int, ...],
    grid_cell_size: float = 5.0,
    grid_cell_offset: float = 0.5,
    dtype: torch.dtype = torch.float,
) -> Tuple[Tensor, Tensor, List[int], Tensor]:
    """
    Like ATSS, generate anchors based on grid size.

    :param feats: shape[s, (b, c, h, w)]
    :param fpn_strides: shape[s], stride for each scale feature
    :param grid_cell_size: anchor size
    :param grid_cell_offset: The range is between 0 and 1.
    :param dtype: Type of the anchors.

    :return:
        - anchors: shape[l, 4], "xmin, ymin, xmax, ymax" format.
        - anchor_points: shape[l, 2], "x, y" format.
        - num_anchors_list: shape[s], contains [s_1, s_2, ...].
        - stride_tensor: shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    device = feats[0].device
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride

        if torch_version_is_greater_or_equal(1, 10):
            # https://github.com/pytorch/pytorch/issues/50276
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        else:
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

        anchor = torch.stack(
            [shift_x - cell_half_size, shift_y - cell_half_size, shift_x + cell_half_size, shift_y + cell_half_size],
            dim=-1,
        ).to(dtype=dtype)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=dtype))

    anchors = torch.cat(anchors).to(device)
    anchor_points = torch.cat(anchor_points).to(device)
    stride_tensor = torch.cat(stride_tensor).to(device)
    return anchors, anchor_points, num_anchors_list, stride_tensor


class ESEAttn(nn.Module):
    def __init__(self, feat_channels: int, activation_type: Type[nn.Module]):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.conv = ConvBNAct(feat_channels, feat_channels, kernel_size=1, padding=0, stride=1, activation_type=activation_type, bias=False)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOEHead(nn.Module):
    @resolve_param("activation", ActivationsTypeFactory())
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        activation: Type[nn.Module] = nn.SiLU,
        fpn_strides: Tuple[int, int, int] = (32, 16, 8),
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        reg_max=16,
        eval_size: Tuple[int, int] = None,
        width_mult: float = 1.0,
    ):
        """

        :param num_classes:
        :param in_channels: Number of channels for each feature map (See width_mult)
        :param activation: Type of the activation used in module
        :param fpn_strides: Output strides of the feature maps from the neck
        :param grid_cell_scale:
        :param grid_cell_offset:
        :param reg_max:
        :param eval_size: (rows, cols) Size of the image for evaluation. Setting this value can be beneficial for inference speed,
               since anchors will not be regenerated for each forward call.
        :param exclude_nms:
        :param exclude_post_process:
        :param width_mult: A scaling factor applied to in_channels in order.
        """
        super(PPYOLOEHead, self).__init__()
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]

        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size

        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()

        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, activation_type=activation))
            self.stem_reg.append(ESEAttn(in_c, activation_type=activation))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, 4 * (self.reg_max + 1), 3, padding=1))

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

        self._init_weights()

    @torch.jit.ignore
    def cache_anchors(self, input_size: Tuple[int, int]):
        self.eval_size = input_size
        anchor_points, stride_tensor = self._generate_anchors()
        self.anchor_points = anchor_points
        self.stride_tensor = stride_tensor

    @torch.jit.ignore
    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            torch.nn.init.constant_(cls_.weight, 0.0)
            torch.nn.init.constant_(cls_.bias, bias_cls)
            torch.nn.init.constant_(reg_.weight, 0.0)
            torch.nn.init.constant_(reg_.bias, 1.0)

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    @torch.jit.ignore
    def replace_num_classes(self, num_classes: int):
        bias_cls = bias_init_with_prob(0.01)
        device = self.pred_cls[0].weight.device
        self.pred_cls = nn.ModuleList()
        self.num_classes = num_classes

        for in_c in self.in_channels:
            predict_layer = nn.Conv2d(in_c, num_classes, 3, padding=1, device=device)
            torch.nn.init.constant_(predict_layer.weight, 0.0)
            torch.nn.init.constant_(predict_layer.bias, bias_cls)
            self.pred_cls.append(predict_layer)

    @torch.jit.ignore
    def forward_train(self, feats: Tuple[Tensor, ...]):
        anchors, anchor_points, num_anchors_list, stride_tensor = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            # Note we don't apply sigmoid on class predictions to ensure good numerical stability at loss computation
            cls_score_list.append(torch.permute(cls_logit.flatten(2), [0, 2, 1]))
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))
        cls_score_list = torch.cat(cls_score_list, dim=1)
        reg_distri_list = torch.cat(reg_distri_list, dim=1)

        return cls_score_list, reg_distri_list, anchors, anchor_points, num_anchors_list, stride_tensor

    def forward_eval(self, feats: Tuple[Tensor, ...]) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, ...]]:

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            avg_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, self.num_classes, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        # Decode bboxes
        # Note in eval mode, anchor_points_inference is different from anchor_points computed on train
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        decoded_predictions = pred_bboxes, pred_scores

        if torch.jit.is_tracing():
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset)

        raw_predictions = cls_score_list, reg_distri_list, anchors, anchor_points, num_anchors_list, stride_tensor
        return decoded_predictions, raw_predictions

    def _generate_anchors(self, feats=None, dtype=torch.float):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            if torch_version_is_greater_or_equal(1, 10):
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            else:
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        if feats is not None:
            anchor_points = anchor_points.to(feats[0].device)
            stride_tensor = stride_tensor.to(feats[0].device)
        return anchor_points, stride_tensor

    def forward(self, feats: Tuple[Tensor]):
        if self.training:
            return self.forward_train(feats)
        else:
            return self.forward_eval(feats)
