"""
ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
(https://arxiv.org/abs/1807.11164)

Code taken from torchvision/models/shufflenetv2.py
"""
from typing import List

import torch
from torch import Tensor
import torch.nn as nn

from super_gradients.training.utils import HpmStruct
from super_gradients.training.models.sg_module import SgModule


__all__ = [
    'ShuffleNetV2Base', 'ShufflenetV2_x0_5', 'ShufflenetV2_x1_0',
    'ShufflenetV2_x1_5', 'ShufflenetV2_x2_0', 'CustomizedShuffleNetV2'
]


class ChannelShuffleInvertedResidual(nn.Module):
    """
    Implement Inverted Residual block as in [https://arxiv.org/abs/1807.11164] in Fig.3 (c) & (d):

    * When stride > 1
      - the whole input goes through branch1,
      - the whole input goes through branch2 ,
      and the arbitrary number of output channels are produced.
    * When stride == 1
      - half of input channels in are passed as identity,
      - another half of input channels goes through branch2,
      and the number of output channels after the block remains the same as in input.

    Channel shuffle is performed on a concatenation in both cases.
    """
    def __init__(self, inp: int, out: int, stride: int) -> None:
        super(ChannelShuffleInvertedResidual, self).__init__()

        assert 1 <= stride <= 3, "Illegal stride value"
        assert (stride != 1) or (inp == out), \
            "When stride == 1 num of input channels should be equal to the requested num of out output channels"

        self.stride = stride
        # half of requested out channels will be produced by each branch
        branch_features = out // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=inp),
                nn.BatchNorm2d(inp),

                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            # won't be called if self.stride == 1
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            # branch 2 operates on the whole input when stride > 1 and on half of it otherwise
            nn.Conv2d(inp if (self.stride > 1) else inp // 2, branch_features, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, bias=False,
                      groups=branch_features),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def channel_shuffle(x: Tensor, groups: int) -> Tensor:
        """
        From "ShuffleNet V2: Practical Guidelines for EfficientCNN Architecture Design" (https://arxiv.org/abs/1807.11164):
            A “channel shuffle” operation is then introduced to enable
            information communication between different groups of channels and improve accuracy.

        The operation preserves x.size(), but shuffles its channels in the manner explained further in the example.

        Example:
            If group = 2 (2 branches with the same # of activation maps were concatenated before channel_shuffle),
            then activation maps in x are:
            from_B1, from_B1, ... from_B2, from_B2
            After channel_shuffle activation maps in x will be:
            from_B1, from_B2, ... from_B1, from_B2
        """

        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            # num channels remains the same due to assert that inp == out in __init__
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # inp num channels can change to a requested arbitrary out num channels
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)
        return out


class ShuffleNetV2Base(SgModule):

    def __init__(self, structure: List[int], stages_out_channels: List[int],
                 backbone_mode: bool = False, num_classes: int = 1000,
                 block: nn.Module = ChannelShuffleInvertedResidual):
        super(ShuffleNetV2Base, self).__init__()

        self.backbone_mode = backbone_mode

        if len(structure) != 3:
            raise ValueError('expected structure as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self.structure = structure
        self.out_channels = stages_out_channels

        input_channels = 3
        output_channels = self.out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.layer2 = self._make_layer(block, input_channels, self.out_channels[1], self.structure[0])
        self.layer3 = self._make_layer(block, self.out_channels[1], self.out_channels[2], self.structure[1])
        self.layer4 = self._make_layer(block, self.out_channels[2], self.out_channels[3], self.structure[2])

        input_channels = self.out_channels[3]
        output_channels = self.out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        if not self.backbone_mode:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(output_channels, num_classes)

    @staticmethod
    def _make_layer(block, input_channels, output_channels, repeats):
        # add first block with stride 2 to downsize the input
        seq = [block(input_channels, output_channels, 2)]

        for _ in range(repeats - 1):
            seq.append(block(output_channels, output_channels, 1))
        return nn.Sequential(*seq)

    def load_state_dict(self, state_dict, strict=True):
        """
        load_state_dict - Overloads the base method and calls it to load a modified dict for usage as a backbone
        :param state_dict:  The state_dict to load
        :param strict:      strict loading (see super() docs)
        """
        pretrained_model_weights_dict = state_dict.copy()

        if self.backbone_mode:
            # removing fc weights first not to break strict loading
            fc_weights_keys = [k for k in pretrained_model_weights_dict if 'fc' in k]

            for key in fc_weights_keys:
                pretrained_model_weights_dict.pop(key)

        super().load_state_dict(pretrained_model_weights_dict, strict)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)

        if not self.backbone_mode:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class ShufflenetV2_x0_5(ShuffleNetV2Base):

    def __init__(self, arch_params: HpmStruct, num_classes: int = 1000, backbone_mode: bool = False):
        super().__init__([4, 8, 4], [24, 48, 96, 192, 1024],
                         backbone_mode=backbone_mode,
                         num_classes=num_classes or arch_params.num_classes)


class ShufflenetV2_x1_0(ShuffleNetV2Base):

    def __init__(self, arch_params: HpmStruct, num_classes: int = 1000, backbone_mode: bool = False):
        super().__init__([4, 8, 4], [24, 116, 232, 464, 1024],
                         backbone_mode=backbone_mode,
                         num_classes=num_classes or arch_params.num_classes)


class ShufflenetV2_x1_5(ShuffleNetV2Base):

    def __init__(self, arch_params: HpmStruct, num_classes: int = 1000, backbone_mode: bool = False):
        super().__init__([4, 8, 4], [24, 176, 352, 704, 1024],
                         backbone_mode=backbone_mode,
                         num_classes=num_classes or arch_params.num_classes)


class ShufflenetV2_x2_0(ShuffleNetV2Base):

    def __init__(self, arch_params: HpmStruct, num_classes: int = 1000, backbone_mode: bool = False):
        super().__init__([4, 8, 4], [24, 244, 488, 976, 2048],
                         backbone_mode=backbone_mode,
                         num_classes=num_classes or arch_params.num_classes)


class CustomizedShuffleNetV2(ShuffleNetV2Base):

    def __init__(self, arch_params: HpmStruct, num_classes: int = 1000, backbone_mode: bool = False):
        super().__init__(arch_params.structure, arch_params.stages_out_channels,
                         backbone_mode=backbone_mode,
                         num_classes=num_classes or arch_params.num_classes)
