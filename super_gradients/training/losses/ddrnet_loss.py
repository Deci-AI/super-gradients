import torch
from typing import Union
from super_gradients.training.losses.ohem_ce_loss import OhemCELoss


class DDRNetLoss(OhemCELoss):
    def __init__(self,
                 threshold: float = 0.7,
                 ohem_percentage: float = 0.1,
                 weights: list = [1.0, 0.4],
                 ignore_label=255,
                 num_pixels_exclude_ignored: bool = False):
        """
        This loss is an extension of the Ohem (Online Hard Example Mining Cross Entropy) Loss.

        as define in paper:
        Accurate Semantic Segmentation of Road Scenes ( https://arxiv.org/pdf/2101.06085.pdf )

        :param threshold: threshold to th hard example mining algorithm
        :param ohem_percentage: minimum percentage of total pixels for the hard example mining algorithm
        (taking only the largest) losses
        :param weights: weights per each input of the loss. This loss supports a multi output (like in DDRNet with
        an auxiliary head). the losses of each head can be weighted.
        :param ignore_label: targets label to be ignored
        :param num_pixels_exclude_ignored: whether to exclude ignore pixels when calculating the mining percentage.
        see OhemCELoss doc for more details.
        """
        super().__init__(threshold=threshold, mining_percent=ohem_percentage, ignore_lb=ignore_label,
                         num_pixels_exclude_ignored=num_pixels_exclude_ignored)
        self.weights = weights

    def forward(self, predictions_list: Union[list, tuple, torch.Tensor],
                targets: torch.Tensor):
        if isinstance(predictions_list, torch.Tensor):
            predictions_list = (predictions_list,)

        assert len(predictions_list) == len(self.weights), "num of prediction must be the same as num of loss weights"

        losses = []
        unweighted_losses = []
        for predictions, weight in zip(predictions_list, self.weights):
            unweighted_loss = super().forward(predictions, targets)
            unweighted_losses.append(unweighted_loss)
            losses.append(unweighted_loss * weight)
        total_loss = sum(losses)
        unweighted_losses.append(total_loss)

        return total_loss, torch.stack(unweighted_losses, dim=0).detach()
