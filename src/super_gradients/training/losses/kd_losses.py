from torch.nn.modules.loss import _Loss, KLDivLoss
import torch

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss


class KDklDivLoss(KLDivLoss):
    """KL divergence wrapper for knowledge distillation"""

    def __init__(self):
        super(KDklDivLoss, self).__init__(reduction="batchmean")

    def forward(self, student_output, teacher_output):
        return super(KDklDivLoss, self).forward(torch.log_softmax(student_output, dim=1), torch.softmax(teacher_output, dim=1))


@register_loss(Losses.KD_LOSS)
class KDLogitsLoss(_Loss):
    """Knowledge distillation loss, wraps the task loss and distillation loss"""

    def __init__(self, task_loss_fn: _Loss, distillation_loss_fn: _Loss = KDklDivLoss(), distillation_loss_coeff: float = 0.5):
        """
        :param task_loss_fn: task loss. E.g., LabelSmoothingCrossEntropyLoss
        :param distillation_loss_fn: distillation loss. E.g., KLDivLoss
        :param distillation_loss_coeff:
        """

        super(KDLogitsLoss, self).__init__()
        self.task_loss_fn = task_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.distillation_loss_coeff = distillation_loss_coeff

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return ["Loss", "Task Loss", "Distillation Loss"]

    def forward(self, kd_module_output, target):
        task_loss = self.task_loss_fn(kd_module_output.student_output, target)
        if isinstance(task_loss, tuple):  # SOME LOSS FUNCTIONS RETURNS LOSS AND LOG_ITEMS
            task_loss = task_loss[0]
        distillation_loss = self.distillation_loss_fn(kd_module_output.student_output, kd_module_output.teacher_output)
        loss = task_loss * (1 - self.distillation_loss_coeff) + distillation_loss * self.distillation_loss_coeff

        return loss, torch.cat((loss.unsqueeze(0), task_loss.unsqueeze(0), distillation_loss.unsqueeze(0))).detach()
