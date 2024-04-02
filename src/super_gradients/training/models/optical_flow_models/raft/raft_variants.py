import copy
from typing import Union, Optional, Callable

import torch
import torch.nn as nn
from omegaconf import DictConfig

from super_gradients.common.object_names import Models

from super_gradients.common.registry import register_model
from super_gradients.module_interfaces import SupportsReplaceInputChannels
from super_gradients.module_interfaces.exportable_optical_flow import ExportableOpticalFlowModel
from super_gradients.training.models import get_arch_params, SgModule
from super_gradients.training.utils.utils import HpmStruct, get_param

from .raft_base import Encoder, ContextEncoder, FlowIterativeBlock


class RAFT(ExportableOpticalFlowModel, SgModule):
    def __init__(self, in_channels, encoder_params, corr_params, flow_params, num_classes):
        super().__init__()

        self.in_channels = in_channels

        self.feature_encoder = Encoder(
            in_channels=self.in_channels,
            in_planes=encoder_params.in_planes,
            output_dim=encoder_params.fnet.output_dim,
            norm_fn=encoder_params.fnet.norm_fn,
            dropout=encoder_params.dropout,
        )

        self.context_encoder = ContextEncoder(
            in_channels=self.in_channels,
            in_planes=encoder_params.in_planes,
            hidden_dim=encoder_params.hidden_dim,
            context_dim=encoder_params.context_dim,
            output_dim=encoder_params.cnet.output_dim,
            norm_fn=encoder_params.cnet.norm_fn,
            dropout=encoder_params.dropout,
        )

        self.flow_iterative_block = FlowIterativeBlock(encoder_params, encoder_params.update_block.hidden_dim, flow_params, corr_params.alternate_corr)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def coords_grid(batch, ht, wd, device):
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = self.coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = self.coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def forward(self, x, **kwargs):
        """Estimate optical flow between pairs of frames"""

        image1 = x[:, 0]
        image2 = x[:, 1]

        # run the feature network
        fmap1, fmap2 = self.feature_encoder([image1, image2])

        # run the context network
        net, inp = self.context_encoder(image1)

        # initialize flow
        coords0, coords1 = self.initialize_flow(image1)

        # run update block network
        flow_predictions, flow_up = self.flow_iterative_block(coords0, coords1, net, inp, fmap1, fmap2)

        if not self.training:
            return flow_up  # removed 1st coords1 - coords0,

        return flow_predictions

    def replace_input_channels(self, in_channels: int, compute_new_weights_fn: Optional[Callable[[nn.Module, int], nn.Module]] = None):
        if isinstance(self.feature_encoder, SupportsReplaceInputChannels) and isinstance(self.context_encoder, SupportsReplaceInputChannels):
            self.feature_encoder.replace_input_channels(in_channels=in_channels, compute_new_weights_fn=compute_new_weights_fn)
            self.context_encoder.replace_input_channels(in_channels=in_channels, compute_new_weights_fn=compute_new_weights_fn)

            self.in_channels = self.get_input_channels()
        else:
            raise NotImplementedError(
                f"`{self.feature_encoder.__class__.__name__}` and `{self.context_encoder.__class__.__name__}` do not support `replace_input_channels`"
            )

    def get_input_channels(self) -> int:
        if isinstance(self.feature_encoder, SupportsReplaceInputChannels) and isinstance(self.context_encoder, SupportsReplaceInputChannels):
            return self.feature_encoder.get_input_channels()
        else:
            raise NotImplementedError(
                f"`{self.feature_encoder.__class__.__name__}` and `{self.context_encoder.__class__.__name__}` do not support `replace_input_channels`"
            )

    def prep_model_for_conversion(self, input_size: Optional[Union[tuple, list]] = None, **kwargs):
        for module in self.modules():
            if module != self and hasattr(module, "prep_model_for_conversion"):
                module.prep_model_for_conversion(input_size, **kwargs)


@register_model(Models.RAFT_S)
class RAFT_S(RAFT):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("raft_s_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            in_channels=merged_arch_params.in_channels,
            encoder_params=merged_arch_params.encoder_params,
            corr_params=merged_arch_params.corr_params,
            flow_params=merged_arch_params.flow_params,
            num_classes=get_param(merged_arch_params, "num_classes", None),
        )


@register_model(Models.RAFT_L)
class RAFT_L(RAFT):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("raft_l_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            in_channels=merged_arch_params.in_channels,
            encoder_params=merged_arch_params.encoder_params,
            corr_params=merged_arch_params.corr_params,
            flow_params=merged_arch_params.flow_params,
            num_classes=get_param(merged_arch_params, "num_classes", None),
        )
