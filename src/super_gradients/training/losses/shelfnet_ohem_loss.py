import torch

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.losses.ohem_ce_loss import OhemCELoss


@register_loss(Losses.SHELFNET_OHEM_LOSS)
class ShelfNetOHEMLoss(OhemCELoss):
    def __init__(self, threshold: float = 0.7, mining_percent: float = 1e-4, ignore_lb: int = 255):
        """
        This loss is an extension of the Ohem (Online Hard Example Mining Cross Entropy) Loss.
        :param threshold: threshold to th hard example mining algorithm
        :param mining_percent: minimum percentage of total pixels for the hard example mining algorithm
        (taking only the largest) losses.
        Default is 1e-4, according to legacy settings, number of 400 pixels for typical input of (512x512) and batch of
         16.
        :param ignore_lb: targets label to be ignored
        """
        super().__init__(threshold=threshold, mining_percent=mining_percent, ignore_lb=ignore_lb)

    def forward(self, predictions_list: list, targets):
        losses = []
        for predictions in predictions_list:
            losses.append(super().forward(predictions, targets))
        total_loss = sum(losses)
        losses.append(total_loss)

        return total_loss, torch.stack(losses, dim=0).detach()

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return ["Loss1/4", "Loss1/8", "Loss1/16", "Loss"]
