import torch

from super_gradients.training.losses.bce_loss import BCE
from super_gradients.training.losses.dice_loss import BinaryDiceLoss


class BCEDiceLoss(torch.nn.Module):
    """
    Binary Cross Entropy + Dice Loss

    Weighted average of BCE and Dice loss

    Attributes:
        loss_weights: list of size 2 s.t loss_weights[0], loss_weights[1] are the weights for BCE, Dice
        respectively.
    """
    def __init__(self, loss_weights=[0.5, 0.5], logits=True):
        super(BCEDiceLoss, self).__init__()
        self.loss_weights = loss_weights
        self.bce = BCE()
        self.dice = BinaryDiceLoss(apply_sigmoid=logits)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        @param input: Network's raw output shaped (N,1,H,W)
        @param target: Ground truth shaped (N,H,W)
        """

        return self.loss_weights[0] * self.bce(input, target) + self.loss_weights[1] * self.dice(input, target)
