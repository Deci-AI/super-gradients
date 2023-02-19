from typing import Union

import torch
from torch.nn.modules.loss import _Loss

from super_gradients.training.models.kd_modules.kd_module import KDOutput


class SegKDLoss(_Loss):
    def __init__(self, kd_loss: _Loss, ce_loss: _Loss, weights: Union[tuple, list] = (1,), kd_loss_weights: Union[tuple, list] = (1, 4)):
        super().__init__()
        self.kd_loss_weights = kd_loss_weights
        self.weights = weights

        self.kd_loss = kd_loss
        self.ce_loss = ce_loss

    def forward(self, preds: KDOutput, target: torch.Tensor):
        assert isinstance(preds, KDOutput)
        teacher_preds = preds.teacher_output
        preds = preds.student_output

        if isinstance(teacher_preds, torch.Tensor):
            teacher_preds = (teacher_preds,)
        if isinstance(preds, torch.Tensor):
            preds = (preds,)

        losses = []
        total_loss = 0
        # Main and auxiliaries feature maps losses
        for i in range(len(preds)):
            ce_loss = self.ce_loss(preds[i], target)
            cwd_loss = self.kd_loss(preds[i], teacher_preds[i], target)

            loss = self.kd_loss_weights[0] * ce_loss + self.kd_loss_weights[1] * cwd_loss
            total_loss += self.weights[i] * loss
            losses += [ce_loss, cwd_loss]

        losses.append(total_loss)

        return total_loss, torch.stack(losses, dim=0).detach()

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        component_names = []
        for i in range(len(self.weights)):
            component_names += [f"Head-{i}_CE_Loss", f"Head-{i}_CWD_Loss"]
        component_names.append("Total_Loss")
        return component_names
