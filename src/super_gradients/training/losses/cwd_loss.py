from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch
from typing import Optional


class CWDKlDivLoss(_Loss):
    def __init__(self, temperature: float = 3.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.T = temperature
        self.ignore_index = ignore_index
        self.kl_div = nn.KLDivLoss(reduction="sum" if ignore_index is None else "none")

    def forward(self, student_preds: torch.Tensor, teacher_preds: torch.Tensor, target: Optional[torch.Tensor] = None):
        B, C, H, W = student_preds.size()
        # Softmax normalization
        softmax_teacher = torch.softmax(teacher_preds.view(B, C, -1) / self.T, dim=-1)
        log_softmax_student = torch.log_softmax(student_preds.view(B, C, -1) / self.T, dim=-1)

        loss = self.kl_div(log_softmax_student, softmax_teacher)

        if self.ignore_index is not None:
            valid_mask = target.view(B, -1).ne(self.ignore_index).unsqueeze(1).expand_as(loss)
            loss = (loss * valid_mask).sum()

        loss = loss * (self.T**2) / (B * C)
        return loss
