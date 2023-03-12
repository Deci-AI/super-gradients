from typing import Union

import torch
import torch.nn as nn

from super_gradients.training.models.kd_modules.kd_module import KDOutput


class SegKDLoss(nn.Module):
    """
    Wrapper loss for semantic segmentation KD.
    This loss includes two loss components, `ce_loss` i.e CrossEntropyLoss, and `kd_loss` i.e
    `ChannelWiseKnowledgeDistillationLoss`.
    """

    def __init__(self, kd_loss: nn.Module, ce_loss: nn.Module, weights: Union[tuple, list], kd_loss_weights: Union[tuple, list]):
        """
        :param kd_loss: knowledge distillation criteria, such as, ChannelWiseKnowledgeDistillationLoss.
         This loss should except as input a triplet of the predictions from the model with shape [B, C, H, W],
         the teacher model predictions with shape [B, C, H, W] and the target labels with shape [B, H, W].
        :param ce_loss: classification criteria, such as, CE, OHEM, MaskAttention, SL1, etc.
         This loss should except as input the predictions from the model with shape [B, C, H, W], and the target labels
         with shape [B, H, W].
        :param weights: lambda weights to apply upon each prediction map heads.
        :param kd_loss_weights: lambda weights to apply upon each criterion. 2 values are excepted as follows,
         [ce_loss_weight, kd_loss_weight].
        """
        super().__init__()
        self.kd_loss_weights = kd_loss_weights
        self.weights = weights

        self.kd_loss = kd_loss
        self.ce_loss = ce_loss

        self._validate_arguments()

    def _validate_arguments(self):
        # Check num of loss weights
        if len(self.kd_loss_weights) != 2:
            raise ValueError(f"kd_loss_weights is expected to be an iterable with size 2," f" found: {len(self.kd_loss_weights)}")

    def forward(self, preds: KDOutput, target: torch.Tensor):
        if not isinstance(preds, KDOutput):
            raise RuntimeError(
                "Predictions argument for `SegKDLoss` forward method is expected to be a `KDOutput` to"
                " include the predictions from both the student and the teacher models."
            )
        teacher_preds = preds.teacher_output
        student_preds = preds.student_output

        if isinstance(teacher_preds, torch.Tensor):
            teacher_preds = (teacher_preds,)
        if isinstance(student_preds, torch.Tensor):
            student_preds = (student_preds,)

        losses = []
        total_loss = 0
        # Main and auxiliaries feature maps losses
        for i in range(len(self.weights)):
            ce_loss = self.ce_loss(student_preds[i], target)
            cwd_loss = self.kd_loss(student_preds[i], teacher_preds[i], target)

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
            component_names += [f"Head-{i}_CE_Loss", f"Head-{i}_KD_Loss"]
        component_names.append("Total_Loss")
        return component_names
