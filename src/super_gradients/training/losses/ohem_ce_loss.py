import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from super_gradients.training.exceptions.loss_exceptions import IllegalRangeForLossAttributeException, RequiredLossComponentReductionException


class OhemLoss(_Loss):
    """
    OhemLoss - Online Hard Example Mining Cross Entropy Loss
    """

    def __init__(self, threshold: float, mining_percent: float = 0.1, ignore_lb: int = -100, num_pixels_exclude_ignored: bool = True, criteria: _Loss = None):
        """
        :param threshold: Sample below probability threshold, is considered hard.
        :param num_pixels_exclude_ignored: How to calculate total pixels from which extract mining percent of the
         samples.
        :param ignore_lb: label index to be ignored in loss calculation.
        :param criteria: loss to mine the examples from.

         i.e for num_pixels=100, ignore_pixels=30, mining_percent=0.1:
         num_pixels_exclude_ignored=False => num_mining = 100 * 0.1 = 10
         num_pixels_exclude_ignored=True  => num_mining = (100 - 30) * 0.1 = 7
        """
        super().__init__()

        if mining_percent < 0 or mining_percent > 1:
            raise IllegalRangeForLossAttributeException((0, 1), "mining percent")

        self.thresh = -torch.log(torch.tensor(threshold, dtype=torch.float))
        self.mining_percent = mining_percent
        self.ignore_lb = ignore_lb
        self.num_pixels_exclude_ignored = num_pixels_exclude_ignored

        if criteria.reduction != "none":
            raise RequiredLossComponentReductionException("criteria", criteria.reduction, "none")
        self.criteria = criteria

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        if self.num_pixels_exclude_ignored:
            # remove ignore label elements
            loss = loss[labels.view(-1) != self.ignore_lb]
            # num pixels in a batch -> num_pixels = batch_size * width * height - ignore_pixels
            num_pixels = loss.numel()
        else:
            num_pixels = labels.numel()
        # if all pixels are ignore labels, return empty loss tensor
        if num_pixels == 0:
            return torch.tensor([0.0]).requires_grad_(True)

        num_mining = int(self.mining_percent * num_pixels)
        # in case mining_percent=1, prevent out of bound exception
        num_mining = min(num_mining, num_pixels - 1)

        self.thresh = self.thresh.to(logits.device)
        loss, _ = torch.sort(loss, descending=True)
        if loss[num_mining] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:num_mining]
        return torch.mean(loss)


class OhemCELoss(OhemLoss):
    """
    OhemLoss - Online Hard Example Mining Cross Entropy Loss
    """

    def __init__(self, threshold: float, mining_percent: float = 0.1, ignore_lb: int = -100, num_pixels_exclude_ignored: bool = True):
        ignore_lb = -100 if ignore_lb is None or ignore_lb < 0 else ignore_lb
        criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction="none")
        super(OhemCELoss, self).__init__(
            threshold=threshold, mining_percent=mining_percent, ignore_lb=ignore_lb, num_pixels_exclude_ignored=num_pixels_exclude_ignored, criteria=criteria
        )


class OhemBCELoss(OhemLoss):
    """
    OhemBCELoss - Online Hard Example Mining Binary Cross Entropy Loss
    """

    def __init__(
        self,
        threshold: float,
        mining_percent: float = 0.1,
        ignore_lb: int = -100,
        num_pixels_exclude_ignored: bool = True,
    ):
        super(OhemBCELoss, self).__init__(
            threshold=threshold,
            mining_percent=mining_percent,
            ignore_lb=ignore_lb,
            num_pixels_exclude_ignored=num_pixels_exclude_ignored,
            criteria=nn.BCEWithLogitsLoss(reduction="none"),
        )

    def forward(self, logits, labels):

        # REMOVE SINGLE CLASS CHANNEL WHEN DEALING WITH BINARY DATA
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return super(OhemBCELoss, self).forward(logits, labels.float())
