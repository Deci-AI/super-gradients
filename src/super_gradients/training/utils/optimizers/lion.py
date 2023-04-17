"""PyTorch implementation of the Lion optimizer.
Code adopted from: https://github.com/google/automl/blob/master/lion/lion_pytorch.py
"""
from typing import Optional, Union, Iterable, Tuple

import torch
from torch.optim.optimizer import Optimizer

from super_gradients.common.object_names import Optimizers
from super_gradients.common.registry.registry import register_optimizer


@register_optimizer(Optimizers.LION)
class Lion(Optimizer):
    r"""Implements Lion algorithm.
    Generaly, it is recommended to divide lr used by AdamW by 10 and multiply the weight decay by 10.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[dict]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        """
        Initialize the hyperparameters.

        :param params:          Iterable of parameters to optimize or dicts defining parameter groups
        :param lr:              Learning rate (default: 1e-4)
        :param betas:           Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.99))
        :param weight_decay:    Weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> torch.Tensor:
        """
        Perform a single optimization step.

        :param closure: A closure that reevaluates the model and returns the loss.
        :return: Loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
