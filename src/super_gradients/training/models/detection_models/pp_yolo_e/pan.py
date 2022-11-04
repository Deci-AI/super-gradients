import collections
from typing import Type, Tuple, List

import torch
from torch import nn, Tensor
from super_gradients.training.models.detection_models.csp_resnet import CSPResNetBasicBlock
from super_gradients.modules import ConvBNAct

__all__ = ["CustomCSPPAN"]


class SPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: Tuple[int, ...],
        activation_type: Type[nn.Module],
    ):
        super().__init__()
        mid_channels = in_channels * (1 + len(pool_size))
        pools = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            pools.append(pool)
        self.pool = nn.ModuleList(pools)
        self.conv = ConvBNAct(mid_channels, out_channels, kernel_size, padding=kernel_size // 2, activation_type=activation_type, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.concat(outs, dim=1)
        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n, activation_type: Type[nn.Module], spp: bool):
        super().__init__()
        ch_mid = int(out_channels // 2)
        self.conv1 = ConvBNAct(in_channels, ch_mid, kernel_size=1, padding=0, activation_type=activation_type, stride=1)
        self.conv2 = ConvBNAct(in_channels, ch_mid, kernel_size=1, padding=0, activation_type=activation_type, stride=1)

        convs = []
        next_ch_in = ch_mid
        for i in range(n):
            convs.append((str(i), CSPResNetBasicBlock(next_ch_in, ch_mid, activation_type=activation_type, use_residual_connection=False)))
            if i == (n - 1) // 2 and spp:
                convs.append(("spp", SPP(ch_mid, ch_mid, 1, (5, 9, 13), activation_type=activation_type)))
            next_ch_in = ch_mid

        self.convs = nn.Sequential(collections.OrderedDict(convs))
        self.conv3 = ConvBNAct(ch_mid * 2, out_channels, kernel_size=1, padding=0, activation_type=activation_type, stride=1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.concat([y1, y2], dim=1)
        y = self.conv3(y)
        return y


class CustomCSPPAN(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, ...] = (256, 512, 1024),
        out_channels: Tuple[int, ...] = (1024, 512, 256),
        activation_type: Type[nn.Module] = nn.SiLU,
        stage_num=1,
        block_num=3,
        block_size=3,
        keep_prob=0.9,
        spp: bool = False,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ):

        super(CustomCSPPAN, self).__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        self.num_blocks = len(in_channels)
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages = []
        fpn_routes = []
        ch_pre = None
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = []
            for j in range(stage_num):
                stage.append(
                    (
                        str(j),
                        CSPStage(
                            ch_in if j == 0 else ch_out,
                            ch_out,
                            block_num,
                            activation_type=activation_type,
                            spp=(spp and i == 0),
                        ),
                    ),
                )

            fpn_stages.append(nn.Sequential(collections.OrderedDict(stage)))

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNAct(
                        in_channels=ch_out,
                        out_channels=ch_out // 2,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        activation_type=activation_type,
                    )
                )

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNAct(
                    in_channels=out_channels[i + 1],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation_type=activation_type,
                )
            )

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = []
            for j in range(stage_num):
                stage.append(
                    (
                        str(j),
                        CSPStage(
                            ch_in if j == 0 else ch_out,
                            ch_out,
                            block_num,
                            activation_type=activation_type,
                            spp=False,
                        ),
                    ),
                )

            pan_stages.append(nn.Sequential(collections.OrderedDict(stage)))

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks: List[Tensor]) -> List[Tensor]:
        blocks = blocks[::-1]
        fpn_feats = []
        route = None
        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.concat([route, block], dim=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = torch.nn.functional.interpolate(route, scale_factor=2, mode="nearest")

        pan_feats = [
            fpn_feats[-1],
        ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.concat([route, block], dim=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]
