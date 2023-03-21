from typing import List
import torch

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.losses.bce_loss import BCE
from super_gradients.training.losses.dice_loss import BinaryDiceLoss


@register_loss(Losses.BCE_DICE_LOSS)
class BCEDiceLoss(torch.nn.Module):
    """
    Binary Cross Entropy + Dice Loss

    Weighted average of BCE and Dice loss

    :param loss_weights: List of size 2 s.t loss_weights[0], loss_weights[1] are the weights for BCE, Dice respectively.
    :param logits:       Whether to use logits or not.
    """

    def __init__(self, loss_weights: List[float] = [0.5, 0.5], logits: bool = True):
        super(BCEDiceLoss, self).__init__()
        self.loss_weights = loss_weights
        self.bce = BCE()
        self.dice = BinaryDiceLoss(apply_sigmoid=logits)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param input: Network's raw output shaped (N,1,H,W)
        :param target: Ground truth shaped (N,H,W)
        """

        return self.loss_weights[0] * self.bce(input, target) + self.loss_weights[1] * self.dice(input, target)
