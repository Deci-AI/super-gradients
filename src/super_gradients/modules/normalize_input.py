from typing import Tuple

import torch
from torch import nn


class NormalizeInput(nn.Module):
    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]):
        super().__init__()
        mean = list(map(float, mean))
        std = list(map(float, std))
        self.register_buffer("mean", torch.tensor(mean).float().reshape(1, len(mean), 1, 1).contiguous())
        self.register_buffer("one_over_std", torch.tensor(std).float().reshape(1, len(std), 1, 1).reciprocal().contiguous())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mean) * self.one_over_std
