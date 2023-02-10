import torch

from super_gradients.modules import ConvBNReLU
from torch import nn


class STDCBlock(nn.Module):
    """
    STDC building block, known as Short Term Dense Concatenate module.
    In STDC module, the kernel size of first block is 1, and the rest of them are simply set as 3.
    """

    def __init__(self, in_channels: int, out_channels: int, steps: int, stdc_downsample_mode: str, stride: int):
        """
        :param steps: The total number of convs in this module, 1 conv 1x1 and (steps - 1) conv3x3.
        :param stdc_downsample_mode: downsample mode in stdc block, supported `avg_pool` for average-pooling and
         `dw_conv` for depthwise-convolution.
        """
        super().__init__()
        assert steps in [2, 3, 4], f"only 2, 3, 4 steps number are supported, found: {steps}"
        self.stride = stride
        self.conv_list = nn.ModuleList()
        # build first step conv 1x1.
        self.conv_list.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, bias=False))
        # build skip connection after first convolution.
        if stride == 1:
            self.skip_step1 = nn.Identity()
        elif stdc_downsample_mode == "avg_pool":
            self.skip_step1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif stdc_downsample_mode == "dw_conv":
            self.skip_step1 = ConvBNReLU(
                out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False, groups=out_channels // 2, use_activation=False
            )
        else:
            raise ValueError(f"stdc_downsample mode is not supported: found {stdc_downsample_mode}," f" must be in [avg_pool, dw_conv]")

        in_channels = out_channels // 2
        mid_channels = in_channels
        # build rest conv3x3 layers.
        for idx in range(1, steps):
            if idx < steps - 1:
                mid_channels //= 2
            conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_list.append(conv)
            in_channels = mid_channels

        # add dw conv before second step for down sample if stride = 2.
        if stride == 2:
            self.conv_list[1] = nn.Sequential(
                ConvBNReLU(
                    out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, groups=out_channels // 2, use_activation=False, bias=False
                ),
                self.conv_list[1],
            )

    def forward(self, x):
        out_list = []
        # run first conv
        x = self.conv_list[0](x)
        out_list.append(self.skip_step1(x))

        for conv in self.conv_list[1:]:
            x = conv(x)
            out_list.append(x)

        out = torch.cat(out_list, dim=1)
        return out
