"""
The loss must be of torch.nn.modules.loss._Loss class.
For commonly used losses, import from deci.core.ADNN.losses

-IMPORTANT: forward(...) should return (loss, loss_items) where loss is the tensor used for backprop (i.e what your
original loss function returns), and loss_items should be a tensor of shape (n_items), of values computed during
the forward pass which we desire to log over the entire epoch. For example- the loss itself should always be logged.
Another examploe is a scenario where the computed loss is the sum of a few components we would like to log- these
entries in loss_items).

-When training, set the "loss_logging_items_names" parameter in train_params to be a list of strings, of length
n_items who's ith element is the name of the ith entry in loss_items. Then each item will be logged, rendered on
tensorboard and "watched" (i.e saving model checkpoints according to it).

-Since running logs will save the loss_items in some internal state, it is recommended that loss_items are detached
from their computational graph for memory efficiency.
"""

import torch.nn as nn
from super_gradients.training.losses.label_smoothing_cross_entropy_loss import cross_entropy


class LabelSmoothingCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    LabelSmoothingCrossEntropyLoss - POC loss class, uses SuperGradient's cross entropy which support distribution as targets.

    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None,
                 from_logits=True):
        super(LabelSmoothingCrossEntropyLoss, self).__init__(weight=weight,
                                                             ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        loss = cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)

        loss_items = loss.detach().unsqueeze(0)

        return loss, loss_items
