from typing import Tuple, Callable, Optional
import einops
import torch
from torch import nn, Tensor

from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import generate_anchors_for_grid_cell
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
        self.cls_pred = nn.Conv2d(bbox_inter_channels, 1 + self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.reg_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(bbox_inter_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(bbox_inter_channels, bbox_inter_channels, kernel_size=3, padding=1, bias=False),
            ]
        )
        self.reg_pred = nn.Conv2d(bbox_inter_channels, 4 * (reg_max + 1), 1, 1, 0)

        self.pose_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, pose_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(pose_inter_channels, pose_inter_channels, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(pose_inter_channels, pose_inter_channels, kernel_size=3, padding=1, bias=False),
            ]
        )
        self.pose_pred = nn.Conv2d(pose_inter_channels, 2 * self.num_classes * (reg_max + 1), kernel_size=1, stride=1, padding=0, bias=True)

        torch.nn.init.zeros_(self.cls_pred.weight)
        torch.nn.init.constant_(self.cls_pred.bias, -4)

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        self.cls_pred = compute_new_weights_fn(self.cls_pred, 1 + num_classes)
        self.pose_pred = compute_new_weights_fn(self.pose_pred, 2 * self.num_classes * (self.reg_max + 1))
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

        pose_logits = cls_output[:, 1:, :, :]
        cls_output = cls_output[:, 0:1, :, :]

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

        pose_output = pose_output.reshape((pose_output.size(0), self.reg_max + 1, self.num_classes, 2, pose_output.size(2), pose_output.size(3)))
        return reg_output, cls_output, pose_logits, pose_output

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
        # Do not apply quantization to this tensor
        bbox_proj_conv = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("bbox_proj_conv", bbox_proj_conv, persistent=False)

        pose_proj_conv = torch.linspace(-self.reg_max // 2, self.reg_max // 2, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("pose_proj_conv", pose_proj_conv, persistent=False)
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
    def _init_weights(self):
        if self.eval_size:
            device = infer_model_device(self)
            dtype = infer_model_dtype(self)

            anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=device)
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    @property
    def out_channels(self):
        return None

    def forward(self, feats: Tuple[Tensor]):
        anchors, anchor_points, num_anchors_list, stride_tensor = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []

        pose_logits_list = []
        pose_regression_list = []
        pose_regression_reduced_list = []

        for i, feat in enumerate(feats):
            feat = self.projections[i](feat)
            b, _, h, w = feat.shape
            reg_distri, cls_logit, pose_logit, pose_outputs = self.head(feat, self.scale_specific_normalization_layers[i])

            # reg_distri: [B, 4 * (self.reg_max + 1), H, W]
            # cls_logit: [B, 1, H, W]
            # pose_logit: [B, self.num_classes, H, W]
            # pose_outputs: [B, self.reg_max + 1, self.num_classes, 2, H, W]
            reg_distri_list.append(einops.rearrange(reg_distri, "b c h w -> b (h w) c"))

            # reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            # reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)
            reg_dist_reduced = einops.rearrange(reg_distri, "b (FOUR bins) h w -> b bins (h w) FOUR", FOUR=4).softmax(dim=1)
            reg_dist_reduced = torch.nn.functional.conv2d(reg_dist_reduced, weight=self.bbox_proj_conv).squeeze(1)
            reg_dist_reduced_list.append(reg_dist_reduced)

            cls_score_list.append(einops.rearrange(cls_logit, "b c h w -> b (h w) c"))

            pose_logits_list.append(einops.rearrange(pose_logit, "b c h w -> b (h w) c"))

            pose_regression_list.append(einops.rearrange(pose_outputs, "b bins c TWO h w -> b (h w) bins c TWO", TWO=2))

            pose_regression_reduced = einops.rearrange(pose_outputs, "b bins c TWO h w -> b bins (c TWO) (h w)", TWO=2).softmax(dim=1)
            pose_regression_reduced = torch.nn.functional.conv2d(pose_regression_reduced, weight=self.pose_proj_conv).squeeze(1)
            pose_regression_reduced_list.append(einops.rearrange(pose_regression_reduced, "b (c TWO) hw -> b hw c TWO", TWO=2, c=self.num_classes))

        cls_score_list = torch.cat(cls_score_list, dim=1)  # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        pose_logits_list = torch.cat(pose_logits_list, dim=1)  # [B, Anchors, Joints]
        pose_regression_list = torch.cat(pose_regression_list, dim=1)  # [B, Anchors, Bins, Num Classes, 2]

        # Decode bboxes
        pred_bbox_scores = cls_score_list.sigmoid()
        pred_bbox_xyxy = batch_distance2bbox(anchors.unsqueeze(0), reg_dist_reduced_list * stride_tensor.unsqueeze(0))  # [B, Anchors, 4]

        # Decode keypoints
        pred_pose_coords = torch.cat(pose_regression_reduced_list, dim=1)  # [B, Anchors, Num Classes, 2]
        pred_pose_coords = pred_pose_coords * stride_tensor.unsqueeze(0).unsqueeze(2) + anchor_points.unsqueeze(0).unsqueeze(2)

        pred_pose_scores = pose_logits_list.sigmoid()  # [B, Anchors, C]

        decoded_predictions = pred_bbox_xyxy, pred_bbox_scores, pred_pose_coords, pred_pose_scores

        if torch.jit.is_tracing():
            return decoded_predictions

        raw_predictions = (
            cls_score_list,
            reg_distri_list,
            pose_logits_list,
            pred_pose_coords,
            pose_regression_list,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
            self.bbox_proj_conv,
            self.pose_proj_conv,
        )
        return decoded_predictions, raw_predictions
