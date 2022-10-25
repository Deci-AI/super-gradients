from enum import Enum
import torch
from typing import Union


class LossReduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"


def apply_reduce(loss: torch.Tensor, reduction: Union[LossReduction, str]):
    if reduction == LossReduction.MEAN.value:
        loss = loss.mean()
    elif reduction == LossReduction.SUM.value:
        loss = loss.sum()
    elif not LossReduction.NONE.value:
        raise ValueError(f"Reduction mode is not supported, expected options are ['mean', 'sum', 'none']" f", found {reduction}")
    return loss
