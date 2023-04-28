from typing import Tuple

import torch.nn.functional
from torch import nn, Tensor

from super_gradients.common.object_names import Losses
from super_gradients.common.registry import register_loss


@register_loss(Losses.RESCORING_LOSS)
class RescoringLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: Tuple[Tensor, Tensor], targets):
        """

        :param predictions: Tuple of (poses, scores)
        :param targets: Target scores
        :return: KD loss between predicted scores and target scores
        """
        return torch.nn.functional.binary_cross_entropy_with_logits(predictions[1], targets)
