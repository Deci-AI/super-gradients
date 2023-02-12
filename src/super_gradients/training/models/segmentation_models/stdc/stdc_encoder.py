from typing import Union, List

from torch import nn

from super_gradients.modules import ConvBNReLU

from super_gradients.training.models.segmentation_models.common import AbstractSegmentationBackbone, FeatureMapOutputSpec
from super_gradients.training.models.segmentation_models.stdc.stdc_block import STDCBlock


class STDCBackbone(AbstractSegmentationBackbone):
    def __init__(
        self,
        block_types: list,
        ch_widths: list,
        num_blocks: list,
        stdc_steps: int = 4,
        stdc_downsample_mode: str = "avg_pool",
        in_channels: int = 3,
        out_down_ratios: Union[tuple, list] = (32,),
    ):
        """
        :param block_types: list of block type for each stage, supported `conv` for ConvBNRelu with 3x3 kernel.
        :param ch_widths: list of output num of channels for each stage.
        :param num_blocks: list of the number of repeating blocks in each stage.
        :param stdc_steps: num of convs steps in each block.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :param in_channels: num channels of the input image.
        :param out_down_ratios: down ratio of output feature maps required from the backbone,
            default (32,) for classification.
        """
        super(STDCBackbone, self).__init__()
        assert len(block_types) == len(ch_widths) == len(num_blocks), (
            f"STDC architecture configuration, block_types, ch_widths, num_blocks, must be defined for the same number"
            f" of stages, found: {len(block_types)} for block_type, {len(ch_widths)} for ch_widths, "
            f"{len(num_blocks)} for num_blocks"
        )

        self.out_widths = []
        self.out_down_ratios = out_down_ratios
        self.stages = nn.ModuleDict()
        self.out_stage_keys = []
        down_ratio = 2
        for block_type, width, blocks in zip(block_types, ch_widths, num_blocks):
            block_name = f"block_s{down_ratio}"
            self.stages[block_name] = self._make_stage(
                in_channels=in_channels,
                out_channels=width,
                block_type=block_type,
                num_blocks=blocks,
                stdc_steps=stdc_steps,
                stdc_downsample_mode=stdc_downsample_mode,
            )
            if down_ratio in out_down_ratios:
                self.out_stage_keys.append(block_name)
                self.out_widths.append(width)
            in_channels = width
            down_ratio *= 2

    def _make_stage(self, in_channels: int, out_channels: int, block_type: str, num_blocks: int, stdc_downsample_mode: str, stdc_steps: int = 4):
        """
        :param in_channels: input channels of stage.
        :param out_channels: output channels of stage.
        :param block_type: stage building block, supported `conv` for 3x3 ConvBNRelu, or `stdc` for STDCBlock.
        :param num_blocks: num of blocks in each stage.
        :param stdc_steps: number of conv3x3 steps in each STDC block, referred as `num blocks` in paper.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        :return: nn.Module
        """
        if block_type == "conv":
            block = ConvBNReLU
            kwargs = {"kernel_size": 3, "padding": 1, "bias": False}
        elif block_type == "stdc":
            block = STDCBlock
            kwargs = {"steps": stdc_steps, "stdc_downsample_mode": stdc_downsample_mode}
        else:
            raise ValueError(f"Block type not supported: {block_type}, excepted: `conv` or `stdc`")

        # first block to apply stride 2.
        blocks = nn.ModuleList([block(in_channels, out_channels, stride=2, **kwargs)])
        # build rest of blocks
        for i in range(num_blocks - 1):
            blocks.append(block(out_channels, out_channels, stride=1, **kwargs))

        return nn.Sequential(*blocks)

    def forward(self, x):
        outputs = []
        for stage_name, stage in self.stages.items():
            x = stage(x)
            if stage_name in self.out_stage_keys:
                outputs.append(x)
        return tuple(outputs)

    def get_backbone_output_spec(self) -> List[FeatureMapOutputSpec]:
        return [FeatureMapOutputSpec(channels=ch, stride=st) for ch, st in zip(self.out_widths, self.out_down_ratios)]


class STDC1Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,)):
        super().__init__(
            block_types=["conv", "conv", "stdc", "stdc", "stdc"],
            ch_widths=[32, 64, 256, 512, 1024],
            num_blocks=[1, 1, 2, 2, 2],
            stdc_steps=4,
            in_channels=in_channels,
            out_down_ratios=out_down_ratios,
        )


class STDC2Backbone(STDCBackbone):
    def __init__(self, in_channels: int = 3, out_down_ratios: Union[tuple, list] = (32,)):
        super().__init__(
            block_types=["conv", "conv", "stdc", "stdc", "stdc"],
            ch_widths=[32, 64, 256, 512, 1024],
            num_blocks=[1, 1, 4, 5, 3],
            stdc_steps=4,
            in_channels=in_channels,
            out_down_ratios=out_down_ratios,
        )
