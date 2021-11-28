"""
This file is used to define the Metrics used for training.
The metrics object must be of torchmetrics.Metric type. For more information on how to use torchmetric.Metric objects and
 implement your own metrics see https://torchmetrics.readthedocs.io/en/latest/pages/overview.html
"""

import torchmetrics
import torch


class Accuracy(torchmetrics.Accuracy):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, top_k=1)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        super().update(preds=preds.softmax(1), target=target)


class Top5(torchmetrics.Accuracy):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step, top_k=5)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        super().update(preds=preds.softmax(1), target=target)
