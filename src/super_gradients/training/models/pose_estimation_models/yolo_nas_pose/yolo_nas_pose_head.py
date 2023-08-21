import math
from typing import Tuple, Union, List, Callable, Optional

import torch
from omegaconf import DictConfig
from torch import nn, Tensor

import super_gradients.common.factories.detection_modules_factory as det_factory
from super_gradients.common.registry import register_detection_module
from super_gradients.modules import ConvBNReLU
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.utils import width_multiplier
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import generate_anchors_for_grid_cell
from super_gradients.training.utils import HpmStruct, torch_version_is_greater_or_equal
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from super_gradients.training.utils.utils import infer_model_dtype, infer_model_device


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

        first_pose_conv = [ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)] if groups else []
        self.pose_convs = nn.Sequential(*first_pose_conv, ConvBNReLU(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.cls_pred = nn.Conv2d(inter_channels, 1, 1, 1, 0)
        self.reg_pred = nn.Conv2d(inter_channels, 4 * (reg_max + 1), 1, 1, 0)
        self.pose_pred = nn.Conv2d(inter_channels, 3 * self.num_classes, 1, 1, 0)  # each keypoint is x,y,confidence

        self.cls_dropout_rate = nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        self.reg_dropout_rate = nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()

        self.grid = torch.zeros(1)
        self.stride = stride

        self.prior_prob = 1e-2
        self._initialize_biases()

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        self.pose_pred = compute_new_weights_fn(self.pose_pred, num_classes)
        self.num_classes = num_classes

    @property
    def out_channels(self):
        return None

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor]:
        """

        :param x: Input feature map of shape [B, Cin, H, W]
        :return: Tuple of [reg_output, cls_output, pose_output]
            - reg_output: Tensor of [B, 4 * (reg_max + 1), H, W]
            - cls_output: Tensor of [B, 1, H, W]
            - pose_output: Tensor of [B, 3 * num_classes, H, W]
        """
        x = self.stem(x)

        cls_feat = self.cls_convs(x)
        cls_feat = self.cls_dropout_rate(cls_feat)
        cls_output = self.cls_pred(cls_feat)

        reg_feat = self.reg_convs(x)
        reg_feat = self.reg_dropout_rate(reg_feat)
        reg_output = self.reg_pred(reg_feat)

        pose_feat = self.pose_convs(x)
        pose_feat = self.reg_dropout_rate(pose_feat)
        pose_output = self.pose_pred(pose_feat)

        return reg_output, cls_output, pose_output

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)


@register_detection_module()
class YoloNASPoseNDFLHeads(BaseDetectionModule, SupportsReplaceNumClasses):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        heads_list: Union[str, HpmStruct, DictConfig],
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
        """
        super(YoloNASPoseNDFLHeads, self).__init__(in_channels)
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]

        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

        self._init_weights()

        factory = det_factory.DetectionModulesFactory()
        heads_list = self._pass_args(heads_list, factory, num_classes, reg_max)

        self.num_heads = len(heads_list)
        fpn_strides: List[int] = []
        for i in range(self.num_heads):
            new_head = factory.get(factory.insert_module_param(heads_list[i], "in_channels", in_channels[i]))
            fpn_strides.append(new_head.stride)
            setattr(self, f"head{i + 1}", new_head)

        self.fpn_strides = tuple(fpn_strides)

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        for i in range(self.num_heads):
            head = getattr(self, f"head{i + 1}")
            head.replace_num_classes(num_classes, compute_new_weights_fn)

        self.num_classes = num_classes

    @staticmethod
    def _pass_args(heads_list, factory, num_classes, reg_max):
        for i in range(len(heads_list)):
            heads_list[i] = factory.insert_module_param(heads_list[i], "num_classes", num_classes)
            heads_list[i] = factory.insert_module_param(heads_list[i], "reg_max", reg_max)
        return heads_list

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

    @torch.jit.ignore
    def forward_train(self, feats: Tuple[Tensor, ...]):
        """

        :param feats:
        :return: Tuple of [cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor]
            - cls_score_list: Tensor of [B, N, 1]
            - reg_distri_list: Tensor of [B, N, 4 * (reg_max + 1)]
            - pose_regression_list: Tensor of [B, N, 3 * C]
            - anchors: Tensor of [N, 4]
            - anchor_points: Tensor of [N, 2]
            - num_anchors_list: List of number of anchors for each feature map
            - stride_tensor: Tensor of [N, 1]
        """
        anchors, anchor_points, num_anchors_list, stride_tensor = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )

        cls_score_list = []
        reg_distri_list = []
        pose_regression_list = []
        for i, feat in enumerate(feats):
            reg_distri, cls_logit, pose_logit = getattr(self, f"head{i + 1}")(feat)
            # cls and reg
            # Note we don't apply sigmoid on class predictions to ensure good numerical stability at loss computation
            cls_score_list.append(torch.permute(cls_logit.flatten(2), [0, 2, 1]))  # [B, C, H, W] -> [B, H * W, C]
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))  # [B, C, H, W] -> [B, H * W, C]
            pose_regression_list.append(torch.permute(pose_logit.flatten(2), [0, 2, 1]))  # [B, C, H, W] -> [B, H * W, C]

        cls_score_list = torch.cat(cls_score_list, dim=1)
        reg_distri_list = torch.cat(reg_distri_list, dim=1)
        pose_regression_list = torch.cat(pose_regression_list, dim=1)
        pose_regression_list = pose_regression_list.reshape(
            (pose_regression_list.size(0), pose_regression_list.size(1), self.num_classes, 3)
        )  # [B, Anchors, C, 3]
        return cls_score_list, reg_distri_list, pose_regression_list, anchors, anchor_points, num_anchors_list, stride_tensor

    def forward_eval(self, feats: Tuple[Tensor, ...]):

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        pose_regression_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit, pose_logit = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, -1, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

            pose_regression_list.append(torch.permute(pose_logit.flatten(2), [0, 2, 1]))  # [B, C, H, W] -> [B, H * W, C]

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        pose_regression_list = torch.cat(pose_regression_list, dim=1)  # [B, Anchors, 3 * C]
        pose_regression_list = pose_regression_list.reshape(
            (pose_regression_list.size(0), pose_regression_list.size(1), self.num_classes, 3)
        )  # [B, Anchors, C, 3]

        pred_pose_coords = pose_regression_list[:, :, :, 0:2]  # [B, Anchors, C, 2]
        pred_pose_scores = pose_regression_list[:, :, :, 2].sigmoid()  # [B, Anchors, C]

        # Decode bboxes
        # Note in eval mode, anchor_points_inference is different from anchor_points computed on train
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        pred_pose_coords = (pred_pose_coords + anchor_points_inference.unsqueeze(0).unsqueeze(2)) * stride_tensor.unsqueeze(0).unsqueeze(2)  # add the grid

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
