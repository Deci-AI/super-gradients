from super_gradients.training.models.sg_module import SgModule
import torch
from super_gradients.training.utils.utils import HpmStruct


class KDModule(SgModule):
    def __init__(self, student: torch.nn.Module, teacher: torch.nn.Module, freeze_teacher_eval_mode: bool = False):
        self.student = student
        self.teacher = teacher
        self.freeze_teacher_eval_mode = freeze_teacher_eval_mode
        self._freeze_teacher()

    def _freeze_teacher(self):
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        self.student.train(mode)
        if not self.freeze_teacher_eval_mode:
            self.teacher.train(mode)

    def eval(self):
        self.student.eval()
        self.teacher.eval()

    def forward(self, x):
        out = {"student_out": self.student(x)}
        with torch.no_grad():
            out["teacher_out"] = self.teacher(x)
        return out

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        if hasattr(self.student, 'initialize_param_groups'):
            # INITIALIZE_PARAM_GROUPS MUST RETURN A LIST OF DICTS WITH 'named_params' AND OPTIMIZER's ATTRIBUTES PER
            # GROUP
            param_groups = self.student.initialize_param_groups(lr, training_params)
        else:
            param_groups = [{'named_params': self.student.named_parameters()}]
        return param_groups

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct,
                            total_batch: int) -> list:
        if hasattr(self.student, "update_param_groups"):
            return self.student.update_param_groups(param_groups, lr, epoch, iter, training_params, total_batch)

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        return self.student.prep_model_for_conversion(input_size, **kwargs)

