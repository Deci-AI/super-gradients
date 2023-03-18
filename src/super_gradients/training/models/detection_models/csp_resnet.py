import collections
import os.path
from pathlib import Path
from typing import List, Type, Tuple, Union, Optional

import torch
from super_gradients.common.registry.registry import register_detection_module
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.activations_type_factory import ActivationsTypeFactory
from torch import nn, Tensor

from super_gradients.modules import RepVGGBlock, EffectiveSEBlock, ConvBNAct

__all__ = ["CSPResNetBackbone", "CSPResNetBasicBlock"]

from super_gradients.training.utils.distributed_training_utils import wait_for_the_master, get_local_rank


class CSPResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation_type: Type[nn.Module], use_residual_connection: bool = True, use_alpha=False):
        """

        :param in_channels:
        :param out_channels:
        :param activation_type:
        :param use_residual_connection: Whether to add input x to the output
        :param use_alpha: If True, enables additional learnable weighting parameter for 1x1 branch in RepVGGBlock
        """
        super().__init__()
        if use_residual_connection and in_channels != out_channels:
            raise RuntimeError(
                f"Number of input channels (got {in_channels}) must be equal to the "
                f"number of output channels (got {out_channels}) when use_residual_connection=True"
            )
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation_type=activation_type, bias=False)
        self.conv2 = RepVGGBlock(
            out_channels, out_channels, activation_type=activation_type, se_type=nn.Identity, use_residual_connection=False, use_alpha=use_alpha
        )
        self.use_residual_connection = use_residual_connection

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.use_residual_connection:
            return x + y
        else:
            return y


class CSPResStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks,
        stride: int,
        activation_type: Type[nn.Module],
        use_attention: bool = True,
        use_alpha: bool = False,
    ):
        """

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param num_blocks: Number of blocks in stage
        :param stride: Desired down-sampling for the stage (Usually 2)
        :param activation_type: Non-linearity type used in child modules.
        :param use_attention: If True, adds EffectiveSEBlock at the end of each stage
        :param use_alpha: If True, enables additional learnable weighting parameter for 1x1 branch in underlying RepVGG blocks (PP-Yolo-E Plus)
        """
        super().__init__()

        mid_channels = (in_channels + out_channels) // 2
        half_mid_channels = mid_channels // 2
        mid_channels = 2 * half_mid_channels

        if stride != 1:
            self.conv_down = ConvBNAct(in_channels, mid_channels, 3, stride=stride, padding=1, activation_type=activation_type, bias=False)
        else:
            self.conv_down = None
        self.conv1 = ConvBNAct(mid_channels, half_mid_channels, kernel_size=1, stride=1, padding=0, activation_type=activation_type, bias=False)
        self.conv2 = ConvBNAct(mid_channels, half_mid_channels, kernel_size=1, stride=1, padding=0, activation_type=activation_type, bias=False)
        self.blocks = nn.Sequential(
            *[
                CSPResNetBasicBlock(
                    in_channels=half_mid_channels,
                    out_channels=half_mid_channels,
                    activation_type=activation_type,
                    use_alpha=use_alpha,
                )
                for _ in range(num_blocks)
            ]
        )
        if use_attention:
            self.attn = EffectiveSEBlock(mid_channels)
        else:
            self.attn = nn.Identity()

        self.conv3 = ConvBNAct(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, activation_type=activation_type, bias=False)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        y = self.attn(y)
        y = self.conv3(y)
        return y


@register_detection_module()
class CSPResNetBackbone(nn.Module):
    """
    CSPResNet backbone
    """

    @resolve_param("activation", ActivationsTypeFactory())
    def __init__(
        self,
        layers: Tuple[int, ...],
        channels: Tuple[int, ...],
        activation: Type[nn.Module],
        return_idx: Tuple[int, int, int],
        use_large_stem: bool,
        width_mult: float,
        depth_mult: float,
        use_alpha: bool,
        pretrained_weights: Optional[str] = None,
        in_channels: int = 3,
    ):
        """

        :param layers: Number of blocks in each stage
        :param channels: Number of channels [stem, stage 0, stage 1, stage 2, ...]
        :param activation: Used activation type for all child modules.
        :param return_idx: Indexes of returned feature maps
        :param use_large_stem: If True, uses 3 conv+bn+act instead of 2 in stem blocks
        :param width_mult: Scaling factor for a number of channels
        :param depth_mult: Scaling factor for a number of blocks in each stage
        :param use_alpha: If True, enables additional learnable weighting parameter for 1x1 branch in RepVGGBlock
        :param pretrained_weights:
        :param in_channels: Number of input channels. Default: 3
        """
        super().__init__()
        channels = [max(round(num_channels * width_mult), 1) for num_channels in channels]
        layers = [max(round(num_layers * depth_mult), 1) for num_layers in layers]

        if use_large_stem:
            self.stem = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "conv1",
                            ConvBNAct(in_channels, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation, bias=False),
                        ),
                        (
                            "conv2",
                            ConvBNAct(
                                channels[0] // 2,
                                channels[0] // 2,
                                3,
                                stride=1,
                                padding=1,
                                activation_type=activation,
                                bias=False,
                            ),
                        ),
                        (
                            "conv3",
                            ConvBNAct(channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation, bias=False),
                        ),
                    ]
                )
            )
        else:
            self.stem = nn.Sequential(
                collections.OrderedDict(
                    [
                        (
                            "conv1",
                            ConvBNAct(3, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation, bias=False),
                        ),
                        (
                            "conv2",
                            ConvBNAct(channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation, bias=False),
                        ),
                    ]
                )
            )

        n = len(channels) - 1
        self.stages = nn.ModuleList(
            [
                CSPResStage(
                    channels[i],
                    channels[i + 1],
                    layers[i],
                    stride=2,
                    activation_type=activation,
                    use_alpha=use_alpha,
                )
                for i in range(n)
            ]
        )

        self._out_channels = channels[1:]
        self._out_strides = [4 * 2**i for i in range(n)]
        self.return_idx = return_idx

        if pretrained_weights:
            if isinstance(pretrained_weights, (str, Path)) and os.path.isfile(str(pretrained_weights)):
                state_dict = torch.load(str(pretrained_weights), map_location="cpu")
            elif isinstance(pretrained_weights, str) and pretrained_weights.startswith("https://"):
                with wait_for_the_master(get_local_rank()):
                    state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
            else:
                raise ValueError("pretrained_weights argument should be a path to local file or url to remote file")
            self.load_state_dict(state_dict)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.fuse_block_residual_branches()

    @property
    def out_channels(self) -> Tuple[int]:
        return tuple(self._out_channels)
