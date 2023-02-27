from typing import Optional

import torch.nn as nn
import torch


class ChannelWiseKnowledgeDistillationLoss(nn.Module):
    """
    Implementation of Channel-wise Knowledge distillation loss.

    paper: "Channel-wise Knowledge Distillation for Dense Prediction", https://arxiv.org/abs/2011.13256
    Official implementation: https://github.com/irfanICMLL/TorchDistiller/tree/main/SemSeg-distill
    """

    def __init__(self, normalization_mode: str = "channel_wise", temperature: float = 4.0, ignore_index: Optional[int] = None):
        """
        :param normalization_mode: default is for `channel-wise` normalization as implemented in the original paper,
         softmax is applied upon the spatial dimensions. For vanilla normalization, to apply softmax upon the channel
         dimension, set this value as `spatial_wise`.
        :param temperature: temperature relaxation value applied upon the logits before the normalization. default value
         is set to `4.0` as the original implementation.
        """
        super().__init__()
        self.T = temperature
        self.ignore_index = ignore_index

        self.kl_div = nn.KLDivLoss(reduction="sum" if ignore_index is None else "none")

        if normalization_mode not in ["channel_wise", "spatial_wise"]:
            raise ValueError(f"Unsupported normalization mode: {normalization_mode}")

        self.normalization_mode = normalization_mode

    def forward(self, student_preds: torch.Tensor, teacher_preds: torch.Tensor, target: Optional[torch.Tensor] = None):
        B, C, H, W = student_preds.size()

        # set the normalization axis and the averaging scalar.
        norm_axis = -1 if self.normalization_mode == "channel_wise" else 1
        averaging_scalar = (B * C) if self.normalization_mode == "channel_wise" else (B * H * W)

        # Softmax normalization
        softmax_teacher = torch.softmax(teacher_preds.view(B, C, -1) / self.T, dim=norm_axis)
        log_softmax_student = torch.log_softmax(student_preds.view(B, C, -1) / self.T, dim=norm_axis)

        loss = self.kl_div(log_softmax_student, softmax_teacher)

        if self.ignore_index is not None:
            valid_mask = target.view(B, -1).ne(self.ignore_index).unsqueeze(1).expand_as(loss)
            loss = (loss * valid_mask).sum()

        loss = loss * (self.T**2) / averaging_scalar
        return loss
