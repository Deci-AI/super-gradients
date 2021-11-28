import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class OhemCELoss(_Loss):
    """
    OhemCELoss - Online Hard Example Mining Cross Entropy Loss
    """
    def __init__(self,
                 threshold: float,
                 mining_percent: float = 0.1,
                 ignore_lb: int = -100,
                 num_pixels_exclude_ignored: bool = True):
        """
        :param threshold: Sample below probability threshold, is considered hard.
        :param num_pixels_exclude_ignored: How to calculate total pixels from which extract mining percent of the
         samples.
         i.e for num_pixels=100, ignore_pixels=30, mining_percent=0.1:
         num_pixels_exclude_ignored=False => num_mining = 100 * 0.1 = 10
         num_pixels_exclude_ignored=True  => num_mining = (100 - 30) * 0.1 = 7
        """
        super().__init__()
        assert 0 <= mining_percent <= 1, "mining percent should be a value from 0 to 1"
        self.thresh = -torch.log(torch.tensor(threshold, dtype=torch.float))
        self.mining_percent = mining_percent
        self.ignore_lb = ignore_lb
        self.num_pixels_exclude_ignored = num_pixels_exclude_ignored
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

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
            return torch.tensor([0.]).requires_grad_(True)

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
