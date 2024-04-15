from typing import Union

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from super_gradients.common.registry import register_loss
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.losses_factory import LossesFactory
from super_gradients.training.losses.loss_utils import apply_reduce, LossReduction


@register_loss()
class OpticalFlowLoss(_Loss):
    @resolve_param("criterion", LossesFactory())
    def __init__(self, criterion: _Loss, gamma: float, max_flow: int = 400, reduction: Union[LossReduction, str] = "mean"):
        """
        Loss function defined over sequence of flow predictions

        :param criterion: The loss criterion,
        :param gamma: Loss weights factor
        :param max_flow: The maximum flow displacement allowed. Flow values above it will be excluded from metric calculation.
        :param reduction: Specifies the reduction to apply to the output: `none` | `mean` | `sum`.
            `none`: no reduction will be applied.
            `mean`: the sum of the output will be divided by the number of elements in the output.
            `sum`: the output will be summed.
            Default: `mean`
        """
        super().__init__()

        self.criterion = criterion
        self.gamma = gamma
        self.max_flow = max_flow
        self.reduction = reduction

    def forward(self, preds: Tensor, target: Tensor):
        flow_loss = 0.0

        flow_gt, valid = target

        if torch.is_tensor(preds):
            preds = [preds]

        # exclude invalid pixels and extremely large displacements
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        n_predictions = len(preds)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = i_weight * (valid[:, None] * self.criterion(preds[i], flow_gt))
            flow_loss += apply_reduce(i_loss, self.reduction)

        return flow_loss
