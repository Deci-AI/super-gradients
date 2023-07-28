import numpy as np
import torch
from torch import nn, Tensor


class CastTensorTo(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.to(self.dtype)

    def float(self):
        self.dtype = torch.float32
        return self

    def half(self):
        self.dtype = torch.float16
        return self


class ApplyMeanStd(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float().reshape((1, -1, 1, 1)), persistent=True)
        self.register_buffer("scale", torch.reciprocal(torch.tensor(std).float()).reshape((1, -1, 1, 1)), persistent=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return (inputs - self.mean) * self.scale


class ChannelSelect(nn.Module):
    def __init__(self, channels: np.ndarray):
        super().__init__()
        self.register_buffer("channels_indexes", torch.tensor(channels).long(), persistent=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs[:, self.channels_indexes, :, :]
