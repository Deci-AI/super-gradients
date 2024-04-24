import dataclasses
from typing import Tuple, Union, List, Callable

import super_gradients.common.factories.detection_modules_factory as det_factory
import torch
from omegaconf import DictConfig
from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.training.utils import HpmStruct, torch_version_is_greater_or_equal
from torch import nn, Tensor


@dataclasses.dataclass
class YoloNASRDecodedPredictions:
    """
    :param boxes_cxcywhr: Tensor of shape [B, Anchors, 5] with predicted boxes in CXCYWHR format
    :param scores: Tensor of shape [B, Anchors, C] with predicted scores for class
    """

    boxes_cxcywhr: Tensor
    scores: Tensor


@dataclasses.dataclass
class YoloNASRLogits:
    """
    :param score_logits: Tensor of shape [B, Anchors, C] with predicted scores for class
    :param size_dist: Tensor of shape [B, Anchors, 2 * (reg_max + 1)] with predicted size distribution.
           Non-multiplied by stride.
    :param size_reduced: Tensor of shape [B, Anchors, 2] with predicted size distribution.
           None-multiplied by stride.
    :param angles: Tensor of shape [B, Anchors, 1] with predicted angles (in radians).
    :param offsets: Tensor of shape [B, Anchors, 2] with predicted offsets.
           Non-multiplied by stride.
    :param anchor_points: Tensor of shape [Anchors, 2] with anchor points.
           Non-multiplied by stride.
    :param strides: Tensor of shape [Anchors] with strides.
    :param reg_max: Number of bins in the regression head
    """

    score_logits: Tensor
    size_dist: Tensor
    size_reduced: Tensor
    angles: Tensor
    offsets: Tensor
    anchor_points: Tensor
    strides: Tensor
    reg_max: int

    def as_decoded(self) -> YoloNASRDecodedPredictions:
        sizes = self.size_reduced * self.strides  # [B, Anchors, 2]
        centers = (self.offsets + self.anchor_points) * self.strides

        return YoloNASRDecodedPredictions(boxes_cxcywhr=torch.cat([centers, sizes, self.angles], dim=-1), scores=self.score_logits.sigmoid())


@register_detection_module()
class YoloNASRNDFLHeads(BaseDetectionModule, SupportsReplaceNumClasses):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        heads_list: List[Union[HpmStruct, DictConfig]],
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        width_mult: float = 1.0,
    ):
        """
        Initializes the YoloNASRNDFLHeads module.

        :param num_classes: Number of detection classes
        :param in_channels: Number of channels for each feature map (See width_mult)
        :param grid_cell_scale: A scaling factor applied to the grid cell coordinates.
               This scaling factor is used to define anchor boxes (see generate_anchors_for_grid_cell).
        :param grid_cell_offset: A fixed offset that is added to the grid cell coordinates.
               This offset represents a 'center' of the cell and is 0.5 by default.
        :param reg_max: Number of bins in the regression head
        :param width_mult: A scaling factor applied to in_channels.

        """
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        super().__init__(in_channels)

        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

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

    def forward(self, feats: Tuple[Tensor, ...]) -> Union[YoloNASRLogits, Tuple[Tensor, Tensor]]:
        """
        Runs the forward for all the underlying heads and concatenate the predictions to a single result.
        :param feats: List of feature maps from the neck of different strides
        :return: In regular eager mode returns YoloNASRLogits dataclass with all the intermediate outputs
                 for model training & evaluation.
                 When in tracing mode, returns a tuple (pred_bboxes, pred_scores) with decoded predictions.
                 pred_bboxes [B, Num Anchors, 5] - Predicted boxes in CXCYWHR format
                 pred_scores [B, Num Anchors, C] - Predicted class probabilities [0..1]

        """

        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        offsets_list = []
        rot_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            reg_output, cls_output, offset_output, rot_output = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_output.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(reg_output.reshape([-1, 2, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])
            reg_dist_reduced = torch.nn.functional.softmax(reg_dist_reduced, dim=1).mul(self.proj_conv).sum(1)

            # cls and reg
            cls_score_list.append(cls_output.reshape([b, -1, height_mul_width]))
            reg_dist_reduced_list.append(reg_dist_reduced)

            offsets_list.append(torch.flatten(offset_output, 2))
            rot_list.append(torch.flatten(rot_output, 2))

        cls_score_list = torch.cat(cls_score_list, dim=-1)  # [B, C, Anchors]
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])  # # [B, Anchors, C]

        offsets_list = torch.cat(offsets_list, dim=-1)
        offsets_list = torch.permute(offsets_list, [0, 2, 1])  # [B, A, 2]

        rot_list = torch.cat(rot_list, dim=-1)
        rot_list = torch.permute(rot_list, [0, 2, 1])  # [B, A, 1]

        reg_distri_list = torch.cat(reg_distri_list, dim=1)  # [B, Anchors, 2 * (self.reg_max + 1)]
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)  # [B, Anchors, 2]

        anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        logits = YoloNASRLogits(
            score_logits=cls_score_list,
            size_dist=reg_distri_list,
            size_reduced=reg_dist_reduced_list,
            offsets=offsets_list,
            anchor_points=anchor_points_inference,
            strides=stride_tensor,
            angles=rot_list,
            reg_max=self.reg_max,
        )

        if torch.jit.is_tracing():
            decoded = logits.as_decoded()
            return decoded.boxes_cxcywhr, decoded.scores

        return logits

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
