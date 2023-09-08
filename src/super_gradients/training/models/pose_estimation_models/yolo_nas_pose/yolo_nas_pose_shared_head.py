from typing import Tuple, Callable, Optional

import torch
from torch import nn, Tensor

from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import generate_anchors_for_grid_cell
from super_gradients.training.utils import torch_version_is_greater_or_equal
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from super_gradients.training.utils.utils import infer_model_dtype, infer_model_device


class YoloNASPoseHead(BaseDetectionModule, SupportsReplaceNumClasses):
    """
    A single level head for shared pose
    """

    def __init__(
        self,
        in_channels: int,
        bbox_inter_channels: int,
        pose_inter_channels: int,
        num_classes: int,
        reg_max: int,
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

        self.reg_max = reg_max
        self.num_classes = num_classes
        self.bbox_inter_channels = bbox_inter_channels
        self.pose_inter_channels = pose_inter_channels

        self.cls_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(bbox_inter_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(bbox_inter_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
            ]
        )
        self.cls_pred = nn.Conv2d(bbox_inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.reg_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(bbox_inter_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(bbox_inter_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
            ]
        )
        self.reg_pred = nn.Conv2d(bbox_inter_channels, 4 * (reg_max + 1), kernel_size=1, stride=1, padding=0, bias=True)

        self.pose_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, pose_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(pose_inter_channels, pose_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(pose_inter_channels, pose_inter_channels, kernel_size=3, padding=1, bias=False),
            ]
        )
        self.pose_pred = nn.Conv2d(pose_inter_channels, 3 * self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)  # each keypoint is x,y,confidence

        self.cls_pred = nn.Conv2d(bbox_inter_channels, 1, 1, 1, 0)
        self.reg_pred = nn.Conv2d(bbox_inter_channels, 4 * (reg_max + 1), 1, 1, 0)
        self.pose_pred = nn.Conv2d(pose_inter_channels, 3 * self.num_classes, 1, 1, 0)  # each keypoint is x,y,confidence
        torch.nn.init.zeros_(self.cls_pred.weight)
        torch.nn.init.constant_(self.cls_pred.bias, -4)

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        self.cls_pred = compute_new_weights_fn(self.pose_pred, 1 + num_classes)
        self.pose_pred = compute_new_weights_fn(self.pose_pred, 2 * num_classes)
        self.num_classes = num_classes

    @property
    def out_channels(self):
        return None

    def forward(self, features, bn_layers) -> Tuple[Tensor, Tensor, Tensor]:
        """

        :param x: Input feature map of shape [B, Cin, H, W]
        :return: Tuple of [reg_output, cls_output, pose_output]
            - reg_output: Tensor of [B, 4 * (reg_max + 1), H, W]
            - cls_output: Tensor of [B, 1, H, W]
            - pose_output: Tensor of [B, num_classes, 3, H, W]
        """
        cls_bn1, cls_bn2, cls_bn3, reg_bn1, reg_bn2, reg_bn3, pose_bn1, pose_bn2, pose_bn3 = bn_layers

        cls_feat = self.cls_convs[0](features)
        cls_feat = torch.nn.functional.relu(cls_bn1(cls_feat), inplace=True)
        cls_feat = self.cls_convs[1](cls_feat)
        cls_feat = torch.nn.functional.relu(cls_bn2(cls_feat), inplace=True)
        cls_feat = self.cls_convs[2](cls_feat)
        cls_feat = torch.nn.functional.relu(cls_bn3(cls_feat), inplace=True)
        cls_output = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs[0](features)
        reg_feat = torch.nn.functional.relu(reg_bn1(reg_feat), inplace=True)
        reg_feat = self.reg_convs[1](reg_feat)
        reg_feat = torch.nn.functional.relu(reg_bn2(reg_feat), inplace=True)
        reg_feat = self.reg_convs[2](reg_feat)
        reg_feat = torch.nn.functional.relu(reg_bn3(reg_feat), inplace=True)
        reg_output = self.reg_pred(reg_feat)

        pose_feat = self.pose_convs[0](features)
        pose_feat = torch.nn.functional.relu(pose_bn1(pose_feat), inplace=True)
        pose_feat = self.pose_convs[1](pose_feat)
        pose_feat = torch.nn.functional.relu(pose_bn2(pose_feat), inplace=True)
        pose_feat = self.pose_convs[2](pose_feat)
        pose_feat = torch.nn.functional.relu(pose_bn3(pose_feat), inplace=True)
        pose_output = self.pose_pred(pose_feat)

        pose_output = pose_output.reshape((pose_output.size(0), self.num_classes, 3, pose_output.size(2), pose_output.size(3)))
        return reg_output, cls_output, pose_output

    def create_normalization_layers(self):
        return [
            # cls
            nn.BatchNorm2d(self.bbox_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            nn.BatchNorm2d(self.bbox_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            nn.BatchNorm2d(self.bbox_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            # reg
            nn.BatchNorm2d(self.bbox_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            nn.BatchNorm2d(self.bbox_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            nn.BatchNorm2d(self.bbox_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            # pose
            nn.BatchNorm2d(self.pose_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            nn.BatchNorm2d(self.pose_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
            nn.BatchNorm2d(self.pose_inter_channels, eps=1e-3, momentum=0.03, affine=True, track_running_stats=True),
        ]


@register_detection_module()
class YoloNASPoseSharedHead(BaseDetectionModule, SupportsReplaceNumClasses):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        inter_channels: int,
        bbox_inter_channels: int,
        pose_inter_channels: int,
        fpn_strides: Tuple[int, int, int],
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        eval_size: Optional[Tuple[int, int]] = None,
        width_mult: float = 1.0,
    ):
        """
        Initializes the NDFLHeads module.

        :param num_classes: Number of detection classes
        :param in_channels: Number of channels for each feature map (See width_mult)
        :param grid_cell_scale:
        :param grid_cell_offset:
        :param reg_max: Number of bins in the regression head
        :param eval_size: (rows, cols) Size of the image for evaluation. Setting this value can be beneficial for inference speed,
               since anchors will not be regenerated for each forward call.
        :param width_mult: A scaling factor applied to in_channels.
        :param pose_offset_multiplier: A scaling factor applied to the pose regression offset. This multiplier is
               meant to reduce absolute magnitude of weights in pose regression layers.
               Default value is 1.0.
        :param compensate_grid_cell_offset: (bool) Controls whether to subtract anchor cell offset from the pose regression.
               If True, predicted pose coordinates decoded as (offsets + anchors - grid_cell_offset) * stride.
               If False, predicted pose coordinates decoded as (offsets + anchors) * stride.
               Default value is True.

        """
        super().__init__(in_channels)
        # in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        inter_channels = max(round(inter_channels * width_mult), 1)
        pose_inter_channels = max(round(pose_inter_channels * width_mult), 1)
        bbox_inter_channels = max(round(bbox_inter_channels * width_mult), 1)

        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size
        self.fpn_strides = tuple(fpn_strides)
        self.num_heads = len(in_channels)

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

        self._init_weights()

        self.head = YoloNASPoseHead(
            in_channels=inter_channels,
            bbox_inter_channels=bbox_inter_channels,
            pose_inter_channels=pose_inter_channels,
            reg_max=reg_max,
            num_classes=num_classes,
        )
        projections = []
        scale_specific_normalization_layers = []

        for i in range(self.num_heads):
            projections.append(nn.Conv2d(in_channels[i], inter_channels, kernel_size=1, stride=1, padding=0, bias=False))
            scale_specific_normalization_layers.append(nn.ModuleList(self.head.create_normalization_layers()))

        self.projections = nn.ModuleList(projections)
        self.scale_specific_normalization_layers = nn.ModuleList(scale_specific_normalization_layers)

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        self.head.replace_num_classes(num_classes, compute_new_weights_fn)
        self.num_classes = num_classes

    @torch.jit.ignore
    def cache_anchors(self, input_size: Tuple[int, int]):
        self.eval_size = input_size
        device = infer_model_device(self)
        dtype = infer_model_dtype(self)

        anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=device)
        self.register_buffer("anchor_points", anchor_points, persistent=False)
        self.register_buffer("stride_tensor", stride_tensor, persistent=False)

    @torch.jit.ignore
    def _init_weights(self):
        if self.eval_size:
            device = infer_model_device(self)
            dtype = infer_model_dtype(self)

            anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=device)
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward_eval(self, feats: Tuple[Tensor, ...]):

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        pose_regression_list = []

        for i, feat in enumerate(feats):
            feat = self.projections[i](feat)
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit, pose_logit = self.head(feat, self.scale_specific_normalization_layers[i])
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, -1, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

            pose_regression_list.append(torch.permute(pose_logit.flatten(3), [0, 3, 1, 2]))  # [B, J, 3, H, W] -> [B, H * W, J, 3]

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        pose_regression_list = torch.cat(pose_regression_list, dim=1)  # [B, Anchors, J, 3]

        # Decode bboxes
        # Note in eval mode, anchor_points_inference is different from anchor_points computed on train
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        # Decode keypoints
        pose_regression_list[:, :, :, 0:2] += anchor_points_inference.unsqueeze(0).unsqueeze(2)
        pose_regression_list[:, :, :, 0:2] -= self.grid_cell_offset
        pose_regression_list[:, :, :, 0:2] *= stride_tensor.unsqueeze(0).unsqueeze(2)

        pred_pose_coords = pose_regression_list[:, :, :, 0:2].detach().clone()  # [B, Anchors, C, 2]
        pred_pose_scores = pose_regression_list[:, :, :, 2].detach().clone().sigmoid()  # [B, Anchors, C]

        decoded_predictions = pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores

        if torch.jit.is_tracing():
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset)

        raw_predictions = cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor
        return decoded_predictions, raw_predictions

    @property
    def out_channels(self):
        return None

    def forward(self, feats: Tuple[Tensor]):
        # if self.training:
        #     return self.forward_train(feats)
        # else:
        return self.forward_eval(feats)

    def _generate_anchors(self, feats=None, dtype=None, device=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []

        dtype = dtype or feats[0].dtype
        device = device or feats[0].device

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

        if device is not None:
            anchor_points = anchor_points.to(device)
            stride_tensor = stride_tensor.to(device)
        return anchor_points, stride_tensor
