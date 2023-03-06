import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiAliasDownsample(nn.Module):
    def __init__(self, in_channels: int, stride: int):
        super().__init__()
        self.kernel_size = 3
        self.stride = stride
        self.channels = in_channels

        a = torch.tensor([1.0, 2.0, 1.0])

        filt = a[:, None] * a[None, :]
        filt = filt / torch.sum(filt)

        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)
