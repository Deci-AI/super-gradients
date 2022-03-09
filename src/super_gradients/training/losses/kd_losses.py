from torch.nn.modules.loss import _Loss, KLDivLoss


class KDLogitsLoss(_Loss):
    """ Knowledge distillation loss, wraps the task loss and distillation loss """
    def __init__(self, task_loss_fn: _Loss, distillation_loss_fn: _Loss = KLDivLoss(), distillation_loss_coeff: float = 0.5):
        '''
        :param task_loss_fn: task loss. E.g., LabelSmoothingCrossEntropyLoss
        :param distillation_loss_fn: disitllation loss. E.g., KLDivLoss
        :param distillation_loss_coeff:
        '''

        super(KDLogitsLoss, self).__init__()
        self.task_loss_fn = task_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.distillation_loss_coeff = distillation_loss_coeff

    def forward(self, kd_module_output, target):
        loss = self.task_loss_fn(kd_module_output['student_out'], target) * (1 - self.distillation_loss_coeff)
        loss += self.distillation_loss_fn(kd_module_output['teacher_out'], kd_module_output['student_out']) * (self.distillation_loss_coeff)
        return loss.mean()


