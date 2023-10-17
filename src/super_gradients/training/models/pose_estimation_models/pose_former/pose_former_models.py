import copy
from typing import Union

import torch
from omegaconf import DictConfig
from torch import nn

from super_gradients.common.factories.detection_modules_factory import DetectionModulesFactory
from super_gradients.common.registry import register_model
from super_gradients.module_interfaces import ExportablePoseEstimationModel, AbstractPoseEstimationDecodingModule
from super_gradients.training.models import SgModule
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.yolo_nas_pose_variants import YoloNASPoseDecodingModule
from super_gradients.training.models.segmentation_models.segformer import MiTBackBone
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils import get_param


def resize_like(x, y):
    return torch.nn.functional.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False)


@register_model()
class PoseFormer(SgModule, ExportablePoseEstimationModel):
    def __init__(
        self,
        backbone: Union[str, dict, HpmStruct, DictConfig],
        neck: Union[str, dict, HpmStruct, DictConfig],
        heads: Union[str, dict, HpmStruct, DictConfig],
        num_classes: int,
        embedding_dim: int = 512,
    ):
        """
        :param num_classes: number of classes
        :param encoder_embed_dims: the patch embedding dimensions (number of output channels in each encoder stage)
        :param encoder_layers: the number of encoder layers in each encoder stage
        :param eff_self_att_reduction_ratio: the reduction ratios of the efficient self-attention in each stage
        :param eff_self_att_heads: number of efficient self-attention heads in each stage
        :param overlap_patch_size:  the patch size of the overlapping patch embedding in each stage
        :param overlap_patch_stride:  the patch stride of the overlapping patch embedding in each stage
        :param overlap_patch_pad:  the patch padding of the overlapping patch embedding in each stage
        :param in_channels:  number of input channels

        """

        super().__init__()

        self.backbone = MiTBackBone(
            embed_dims=backbone["encoder_embed_dims"],
            encoder_layers=backbone["encoder_layers"],
            eff_self_att_reduction_ratio=backbone["eff_self_att_reduction_ratio"],
            eff_self_att_heads=backbone["eff_self_att_heads"],
            overlap_patch_size=backbone["overlap_patch_size"],
            overlap_patch_stride=backbone["overlap_patch_stride"],
            overlap_patch_pad=backbone["overlap_patch_pad"],
            in_channels=backbone["in_channels"],
        )

        factory = DetectionModulesFactory()
        self.neck = factory.get(factory.insert_module_param(neck, "in_channels", tuple(backbone["encoder_embed_dims"])))
        self.heads = factory.get(factory.insert_module_param(heads, "in_channels", self.neck.out_channels))

        # self.head = factory.get(factory.insert_module_param(heads, "in_channels", (embedding_dim, embedding_dim, embedding_dim)))

        # self.init_params()

        self.num_classes = num_classes

        # self.use_sliding_window_validation = False
        # input_channels = sum(backbone["encoder_embed_dims"][1:])

        # self.f1_path = ConvBNAct(input_channels, embedding_dim, kernel_size=3, padding=1, bias=False, activation_type=nn.GELU)
        # self.f2_path = ConvBNAct(input_channels, embedding_dim, kernel_size=3, padding=1, bias=False, activation_type=nn.GELU)
        # self.f3_path = ConvBNAct(input_channels, embedding_dim, kernel_size=3, padding=1, bias=False, activation_type=nn.GELU)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.neck(features)
        out = self.heads(features)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        Custom param groups for training:
        - Different lr for backbone and the rest, if `multiply_head_lr` key is in `training_params`.
        """
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        multiply_lr_params, no_multiply_params = self._separate_lr_multiply_params()
        param_groups = [
            {"named_params": no_multiply_params, "lr": lr, "name": "no_multiply_params"},
            {"named_params": multiply_lr_params, "lr": lr * multiply_head_lr, "name": "multiply_lr_params"},
        ]
        return param_groups

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        for param_group in param_groups:
            param_group["lr"] = lr
            if param_group["name"] == "multiply_lr_params":
                param_group["lr"] *= multiply_head_lr
        return param_groups

    def _separate_lr_multiply_params(self):
        """
        Separate backbone params from the rest.
        :return: iterators of groups named_parameters.
        """
        backbone_names = [n for n, p in self.backbone.named_parameters()]
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if name in backbone_names:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()

    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractPoseEstimationDecodingModule:
        return YoloNASPoseDecodingModule(num_pre_nms_predictions)


@register_model()
class PoseFormer_B2(PoseFormer):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        from super_gradients.training.models import get_arch_params

        default_arch_params = get_arch_params("pose_former_b2_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.head,
            num_classes=get_param(merged_arch_params, "num_classes", None),
        )


@register_model()
class PoseFormer_B5(PoseFormer):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        from super_gradients.training.models import get_arch_params

        default_arch_params = get_arch_params("pose_former_b5_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.head,
            num_classes=get_param(merged_arch_params, "num_classes", None),
        )
