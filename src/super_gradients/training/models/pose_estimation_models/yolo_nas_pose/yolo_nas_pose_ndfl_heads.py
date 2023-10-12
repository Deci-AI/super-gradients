from typing import Tuple, Union, List, Callable, Optional

import torch
from omegaconf import DictConfig
from torch import nn, Tensor

import super_gradients.common.factories.detection_modules_factory as det_factory
from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import generate_anchors_for_grid_cell
from super_gradients.training.utils import HpmStruct, torch_version_is_greater_or_equal
from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from super_gradients.training.utils.utils import infer_model_dtype, infer_model_device

# Declare type aliases for better readability
# We cannot use typing.TypeAlias since it is not supported in python 3.7
YoloNasPoseDecodedPredictions = Tuple[Tensor, Tensor, Tensor, Tensor]
YoloNasPoseRawOutputs = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[int], Tensor]


@register_detection_module()
class YoloNASPoseNDFLHeads(BaseDetectionModule, SupportsReplaceNumClasses):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        heads_list: List[Union[HpmStruct, DictConfig]],
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        inference_mode: bool = False,
        eval_size: Optional[Tuple[int, int]] = None,
        width_mult: float = 1.0,
        pose_offset_multiplier: float = 1.0,
        compensate_grid_cell_offset: bool = True,
    ):
        """
        Initializes the NDFLHeads module.

        :param num_classes: Number of detection classes
        :param in_channels: Number of channels for each feature map (See width_mult)
        :param grid_cell_scale: A scaling factor applied to the grid cell coordinates.
               This scaling factor is used to define anchor boxes (see generate_anchors_for_grid_cell).
        :param grid_cell_offset: A fixed offset that is added to the grid cell coordinates.
               This offset represents a 'center' of the cell and is 0.5 by default.
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
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        super().__init__(in_channels)

        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size
        self.pose_offset_multiplier = pose_offset_multiplier
        self.compensate_grid_cell_offset = compensate_grid_cell_offset
        self.inference_mode = inference_mode

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

        self._init_weights()

        factory = det_factory.DetectionModulesFactory()
        heads_list = self._insert_heads_list_params(heads_list, factory, num_classes, reg_max)

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
    def _insert_heads_list_params(
        heads_list: List[Union[HpmStruct, DictConfig]], factory: det_factory.DetectionModulesFactory, num_classes: int, reg_max: int
    ) -> List[Union[HpmStruct, DictConfig]]:
        """
        Injects num_classes and reg_max parameters into the heads_list.

        :param heads_list:  Input heads list
        :param factory:     DetectionModulesFactory
        :param num_classes: Number of classes
        :param reg_max:     Number of bins in the regression head
        :return:            Heads list with injected parameters
        """
        for i in range(len(heads_list)):
            heads_list[i] = factory.insert_module_param(heads_list[i], "num_classes", num_classes)
            heads_list[i] = factory.insert_module_param(heads_list[i], "reg_max", reg_max)
        return heads_list

    @torch.jit.ignore
    def _init_weights(self):
        if self.eval_size:
            device = infer_model_device(self)
            dtype = infer_model_dtype(self)

            anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=device)
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward(self, feats: Tuple[Tensor, ...]) -> Union[YoloNasPoseDecodedPredictions, Tuple[YoloNasPoseDecodedPredictions, YoloNasPoseRawOutputs]]:
        """
        Runs the forward for all the underlying heads and concatenate the predictions to a single result.
        :param feats: List of feature maps from the neck of different strides
        :return: Return value depends on the mode:
        If tracing, a tuple of 4 tensors (decoded predictions) is returned:
        - pred_bboxes [B, Num Anchors, 4] - Predicted boxes in XYXY format
        - pred_scores [B, Num Anchors, 1] - Predicted scores for each box
        - pred_pose_coords [B, Num Anchors, Num Keypoints, 2] - Predicted poses in XY format
        - pred_pose_scores [B, Num Anchors, Num Keypoints] - Predicted scores for each keypoint

        In training/eval mode, a tuple of 2 tensors returned:
        - decoded predictions - they are the same as in tracing mode
        - raw outputs - a tuple of 8 elements in total, this is needed for training the model.
        """

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        pose_regression_list = []
        pose_logits_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_distri, cls_logit, pose_regression, pose_logits = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_distri.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist_reduced, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            cls_score_list.append(cls_logit.reshape([b, -1, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

            pose_regression_list.append(torch.permute(pose_regression.flatten(3), [0, 3, 1, 2]))  # [B, J, 2, H, W] -> [B, H * W, J, 2]
            pose_logits_list.append(torch.permute(pose_logits.flatten(2), [0, 2, 1]))  # [B, J, H, W] -> [B, H * W, J]

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 4 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 4]

        pose_regression_list = torch.cat(pose_regression_list, dim=1)  # [B, Anchors, J, 2]
        pose_logits_list = torch.cat(pose_logits_list, dim=1)  # [B, Anchors, J]

        # Decode bboxes
        # Note in eval mode, anchor_points_inference is different from anchor_points computed on train
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor  # [B, Anchors, 4]

        # Decode keypoints
        if self.pose_offset_multiplier != 1.0:
            pose_regression_list *= self.pose_offset_multiplier

        if self.compensate_grid_cell_offset:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2) - self.grid_cell_offset
        else:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2)

        pose_regression_list *= stride_tensor.unsqueeze(0).unsqueeze(2)

        pred_pose_coords = pose_regression_list.detach().clone()  # [B, Anchors, C, 2]
        pred_pose_scores = pose_logits_list.detach().clone().sigmoid()  # [B, Anchors, C]

        decoded_predictions = pred_bboxes, pred_scores, pred_pose_coords, pred_pose_scores

        if torch.jit.is_tracing() or self.inference_mode:
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset)

        raw_predictions = cls_score_list, reg_distri_list, pose_regression_list, pose_logits_list, anchors, anchor_points, num_anchors_list, stride_tensor
        return decoded_predictions, raw_predictions

    @property
    def out_channels(self):
        return None

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
