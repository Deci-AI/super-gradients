from super_gradients.training.models.sg_module import SgModule
import torch
from super_gradients.training.utils.utils import HpmStruct
from typing import Union
from super_gradients.training.models.all_architectures import ARCHITECTURES
from super_gradients.training.utils.checkpoint_utils import load_pretrained_weights
from super_gradients.training.utils import get_param


class KDOutput:
    def __init__(self, student_output: torch.Tensor = None, teacher_output: torch.Tensor = None):
        self.student_output = student_output
        self.teacher_output = teacher_output


class KDModule(SgModule):
    def __init__(self, student: Union[torch.nn.Module, str], teacher: Union[torch.nn.Module, str],
                 freeze_teacher_eval_mode: bool = False, student_arch_params: dict = {}, teacher_arch_params: dict = {},
                 teacher_ckpt_path: str = None):
        self.student = student
        self.student_arch_params = student_arch_params
        self.teacher_arch_params = teacher_arch_params
        self.teacher = teacher
        self.freeze_teacher_eval_mode = freeze_teacher_eval_mode
        self._freeze_teacher()

    # def _initialize_net(self, net, arch_params, ckpt_path=None):
    #     if isinstance(net, str):
    #         architecture = net
    #         architecture_cls = ARCHITECTURES[architecture]
    #         net_module = architecture_cls(arch_params=self.arch_params)
    #         pretrained_weights = get_param(arch_params, "pretrained_weights")
    #         if pretrained_weights and ckpt_path:
    #             raise ValueError("Expected atmost one of: pretrained_weights, and ckpt_path for teacher module.")
    #         elif pretrained_weights:
    #             load_pretrained_weights(net_module, architecture, pretrained_weights)
    #
    #     elif not isinstance(net, torch.nn.Module):
    #         raise TypeError("Student and teacher networks expected to be str, or torch.nn.Module.")
    #
    #         load_pretrained_weights(net)

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
        return KDOutput(student_output=self.student(x),
                        teacher_output=self.teacher(x))

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
        return self.student.update_param_groups(param_groups, lr, epoch, iter, training_params, total_batch)
