import torch
from torch.nn import BCEWithLogitsLoss


class BCE(BCEWithLogitsLoss):
    """
    Binary Cross Entropy Loss
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param input: Network's raw output shaped (N,1,*)
        :param target: Ground truth shaped (N,*)
        """
        return super(BCE, self).forward(input.squeeze(1), target.float())
