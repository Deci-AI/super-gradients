from super_gradients.training.models.sg_module import SgModule
from collections import namedtuple
import torch

from super_gradients.common.registry.registry import register_kd_model, register_model
from super_gradients.common.object_names import Models
from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.utils import get_param

KDOutput = namedtuple("KDOutput", "student_output teacher_output")


@register_model(Models.KD_MODULE)
@register_kd_model(Models.KD_MODULE)
class KDModule(SgModule):
    """
    KDModule

    class implementing Knowledge Distillation logic as an SgModule

    attributes:
        student: SgModule - the student model
        teacher: torch.nn.Module- the teacher model
        run_teacher_on_eval: bool- whether to run self.teacher at eval mode regardless of self.train(mode)
        arch_params: HpmStruct- Architecture H.P.

            Additionally, by passing teacher_input_adapter (torch.nn.Module) one can modify the teacher net to act as if
            teacher = torch.nn.Sequential(teacher_input_adapter, teacher). This is useful when teacher net expects a
            different input format from the student (for example different normalization).
            Equivalent arg for the student model, can be passed through student_input_adapter.

    """

    def __init__(self, arch_params: HpmStruct, student: SgModule, teacher: torch.nn.Module, run_teacher_on_eval=False):
        super(KDModule, self).__init__()
        self.arch_params = arch_params
        self.student = student
        self.teacher = teacher
        self.teacher_input_adapter = get_param(self.arch_params, "teacher_input_adapter")
        self.student_input_adapter = get_param(self.arch_params, "student_input_adapter")
        self.run_teacher_on_eval = run_teacher_on_eval
        self._freeze_teacher()

        # WHEN CREATING A MODULE SELF.TRAIN() ISN'T CALLED AND SO THE TEACHER MUST BE MOVED TO EVAL MODE EXPLICITLY
        if self.run_teacher_on_eval:
            self.teacher.eval()

    def _freeze_teacher(self):
        for p in self.teacher.parameters():
            p.requires_grad = False

        if self.teacher_input_adapter is not None:
            for p in self.teacher_input_adapter.parameters():
                p.requires_grad = False
            self.teacher_input_adapter.eval()

    def train(self, mode=True):
        self.student.train(mode)
        if not self.run_teacher_on_eval:
            self.teacher.train(mode)

    def eval(self):
        self.student.eval()
        self.teacher.eval()

    def forward(self, x):
        if self.student_input_adapter is not None:
            student_output = self.student(self.student_input_adapter(x))
        else:
            student_output = self.student(x)

        if self.teacher_input_adapter is not None:
            teacher_output = self.teacher(self.teacher_input_adapter(x))
        else:
            teacher_output = self.teacher(x)

        return KDOutput(student_output=student_output, teacher_output=teacher_output)

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        return self.student.initialize_param_groups(lr, training_params)

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        return self.student.update_param_groups(param_groups, lr, epoch, iter, training_params, total_batch)

    def replace_head(self, **kwargs):
        self.student.replace_head(**kwargs)
