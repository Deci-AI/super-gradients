from functools import partial
from typing import Type, List

import torch
from torch import nn, Tensor

from super_gradients.common.registry import register_detection_module
from super_gradients.modules import Residual, BaseDetectionModule
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.activations_type_factory import ActivationsTypeFactory
from super_gradients.modules import QARepVGGBlock, Conv
from super_gradients.modules.utils import width_multiplier

__all__ = ["YoloNASStage", "YoloNASUpStage", "YoloNASStem", "YoloNASDownStage", "YoloNASBottleneck"]


class YoloNASBottleneck(nn.Module):
    """
    A bottleneck block for YoloNAS. Consists of two consecutive blocks and optional residual connection.
    """

    def __init__(
        self, input_channels: int, output_channels: int, block_type: Type[nn.Module], activation_type: Type[nn.Module], shortcut: bool, use_alpha: bool
    ):
        """
        Initialize the YoloNASBottleneck block

        :param input_channels: Number of input channels
        :param output_channels: Number of output channels
        :param block_type: Type of the convolutional block
        :param activation_type: Activation type for the convolutional block
        :param shortcut: If True, adds the residual connection from input to output.
        :param use_alpha: If True, adds the learnable alpha parameter (multiplier for the residual connection).
        """
        super().__init__()

        self.cv1 = block_type(input_channels, output_channels, activation_type=activation_type)
        self.cv2 = block_type(output_channels, output_channels, activation_type=activation_type)
        self.add = shortcut and input_channels == output_channels
        self.shortcut = Residual() if self.add else None
        if use_alpha:
            self.alpha = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.alpha = 1.0

    def forward(self, x):
        return self.alpha * self.shortcut(x) + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SequentialWithIntermediates(nn.Sequential):
    """
    A Sequential module that can return all intermediate values as a list of Tensors
    """

    def __init__(self, output_intermediates: bool, *args):
        super(SequentialWithIntermediates, self).__init__(*args)
        self.output_intermediates = output_intermediates

    def forward(self, input: Tensor) -> List[Tensor]:
        if self.output_intermediates:
            output = [input]
            for module in self:
                output.append(module(output[-1]))
            return output
        #  For uniformity, we return a list even if we don't output intermediates
        return [super(SequentialWithIntermediates, self).forward(input)]


class YoloNASCSPLayer(nn.Module):
    """
    Cross-stage layer module for YoloNAS.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int,
        block_type: Type[nn.Module],
        activation_type: Type[nn.Module],
        shortcut: bool = True,
        use_alpha: bool = True,
        expansion: float = 0.5,
        hidden_channels: int = None,
        concat_intermediates: bool = False,
    ):
        """

        :param in_channels: Number of input channels.
        :param out_channels:  Number of output channels.
        :param num_bottlenecks: Number of bottleneck blocks.
        :param block_type: Bottleneck block type.
        :param activation_type: Activation type for all blocks.
        :param shortcut: If True, adds the residual connection from input to output.
        :param use_alpha: If True, adds the learnable alpha parameter (multiplier for the residual connection).
        :param expansion: If hidden_channels is None, hidden_channels is set to in_channels * expansion.
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks.
        :param concat_intermediates:
        """
        super(YoloNASCSPLayer, self).__init__()
        if hidden_channels is None:
            hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv2 = Conv(in_channels, hidden_channels, 1, stride=1, activation_type=activation_type)
        self.conv3 = Conv(hidden_channels * (2 + concat_intermediates * num_bottlenecks), out_channels, 1, stride=1, activation_type=activation_type)
        module_list = [YoloNASBottleneck(hidden_channels, hidden_channels, block_type, activation_type, shortcut, use_alpha) for _ in range(num_bottlenecks)]
        self.bottlenecks = SequentialWithIntermediates(concat_intermediates, *module_list)

    def forward(self, x: Tensor) -> Tensor:
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((*x_1, x_2), dim=1)
        return self.conv3(x)


@register_detection_module()
class YoloNASStem(BaseDetectionModule):
    """
    Stem module for YoloNAS. Consists of a single QARepVGGBlock with stride of two.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the YoloNASStem module
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super().__init__(in_channels)
        self._out_channels = out_channels
        self.conv = QARepVGGBlock(in_channels, out_channels, stride=2, use_residual_connection=False)

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


@register_detection_module()
class YoloNASStage(BaseDetectionModule):
    """
    A single stage module for YoloNAS. It consists of a downsample block (QARepVGGBlock) followed by YoloNASCSPLayer.
    """

    @resolve_param("activation_type", ActivationsTypeFactory())
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        activation_type: Type[nn.Module],
        hidden_channels: int = None,
        concat_intermediates: bool = False,
    ):
        """
        Initialize the YoloNASStage module
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param num_blocks: Number of bottleneck blocks in the YoloNASCSPLayer
        :param activation_type: Activation type for all blocks
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks.
        :param concat_intermediates: If True, concatenates the intermediate values from the YoloNASCSPLayer.
        """
        super().__init__(in_channels)
        self._out_channels = out_channels
        self.downsample = QARepVGGBlock(in_channels, out_channels, stride=2, activation_type=activation_type, use_residual_connection=False)
        self.blocks = YoloNASCSPLayer(
            out_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            True,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.blocks(self.downsample(x))


@register_detection_module()
class YoloNASUpStage(BaseDetectionModule):
    """
    Upsampling stage for YoloNAS.
    """

    @resolve_param("activation_type", ActivationsTypeFactory())
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module],
        hidden_channels: int = None,
        concat_intermediates: bool = False,
        reduce_channels: bool = False,
    ):
        """
        Initialize the YoloNASUpStage module
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param width_mult: Multiplier for the number of channels in the stage.
        :param num_blocks: Number of bottleneck blocks
        :param depth_mult: Multiplier for the number of blocks in the stage.
        :param activation_type: Activation type for all blocks
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks
        :param concat_intermediates:
        :param reduce_channels:
        """
        super().__init__(in_channels)

        num_inputs = len(in_channels)
        if num_inputs == 2:
            in_channels, skip_in_channels = in_channels
        else:
            in_channels, skip_in_channels1, skip_in_channels2 = in_channels
            skip_in_channels = skip_in_channels1 + out_channels  # skip2 downsample results in out_channels channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        if num_inputs == 2:
            self.reduce_skip = Conv(skip_in_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
        else:
            self.reduce_skip1 = Conv(skip_in_channels1, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()
            self.reduce_skip2 = Conv(skip_in_channels2, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        self.conv = Conv(in_channels, out_channels, 1, 1, activation_type)
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
        if num_inputs == 3:
            self.downsample = Conv(out_channels if reduce_channels else skip_in_channels2, out_channels, kernel=3, stride=2, activation_type=activation_type)

        self.reduce_after_concat = Conv(num_inputs * out_channels, out_channels, 1, 1, activation_type) if reduce_channels else nn.Identity()

        after_concat_channels = out_channels if reduce_channels else out_channels + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
        )

        self._out_channels = [out_channels, out_channels]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        if len(inputs) == 2:
            x, skip_x = inputs
            skip_x = [self.reduce_skip(skip_x)]
        else:
            x, skip_x1, skip_x2 = inputs
            skip_x1, skip_x2 = self.reduce_skip1(skip_x1), self.reduce_skip2(skip_x2)
            skip_x = [skip_x1, self.downsample(skip_x2)]
        x_inter = self.conv(x)
        x = self.upsample(x_inter)
        x = torch.cat([x, *skip_x], 1)
        x = self.reduce_after_concat(x)
        x = self.blocks(x)
        return x_inter, x


@register_detection_module()
class YoloNASDownStage(BaseDetectionModule):
    @resolve_param("activation_type", ActivationsTypeFactory())
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: Type[nn.Module],
        hidden_channels: int = None,
        concat_intermediates: bool = False,
    ):
        """
        Initializes a YoloNASDownStage.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param width_mult: Multiplier for the number of channels in the stage.
        :param num_blocks: Number of blocks in the stage.
        :param depth_mult: Multiplier for the number of blocks in the stage.
        :param activation_type: Type of activation to use inside the blocks.
        :param hidden_channels: If not None, sets the number of hidden channels used inside the bottleneck blocks.
        :param concat_intermediates:
        """

        super().__init__(in_channels)

        in_channels, skip_in_channels = in_channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks

        self.conv = Conv(in_channels, out_channels // 2, 3, 2, activation_type)
        after_concat_channels = out_channels // 2 + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            in_channels=after_concat_channels,
            out_channels=out_channels,
            num_bottlenecks=num_blocks,
            block_type=partial(Conv, kernel=3, stride=1),
            activation_type=activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
        )

        self._out_channels = out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.conv(x)
        x = torch.cat([x, skip_x], 1)
        x = self.blocks(x)
        return x
