import super_gradients.training.sg_model.sg_model
from super_gradients.training.models.all_architectures import KD_ARCHITECTURES
from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.sg_model import SgModel
from typing import Union
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import utils as core_utils
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import get_param
from super_gradients.training.models.sg_module import SgModule
import torch

logger = get_logger(__name__)


class KDModel(SgModel):
    """
    KDModel

    This class extends SgModel to support knowledge distillation.
    """

    def build_model(self,
                    # noqa: C901 - too complex
                    architecture: Union[str, KDModule] = 'kd_module',
                    arch_params={}, checkpoint_params={},
                    *args, **kwargs):

        # FIXME: STUDENTAND TEACHER ARCHITECTURE ARE ONLY PASSED THROUGH KWARGS
        """
        :param architecture: (Union[str, KDModule]) Defines the network's architecture from models/KD_ARCHITECTURES
         (default='kd_module')

        :param arch_params: (dict) Architecture H.P. e.g.: block, num_blocks, num_classes, etc to be passed to kd
            architecture class (discarded when architecture is KDModule instance)

        :param checkpoint_params: (dict) A dictionary like object with the following keys/values:

              student_pretrained_weights:   String describing the dataset of the pretrained weights (for example
              "imagenent") for the student network.

              teacher_pretrained_weights:   String describing the dataset of the pretrained weights (for example
              "imagenent") for the teacher network.

              teacher_checkpoint_path:    Local path to the teacher's checkpoint. Note that when passing pretrained_weights
                                   through teacher_arch_params these weights will be overridden by the
                                   pretrained checkpoint. (default=None)

              load_kd_model_checkpoint:   Whether to load an entire KDModule checkpoint (used to continue KD training)
               (default=False)

              kd_model_source_ckpt_folder_name: Folder name to load an entire KDModule checkpoint from
                (self.experiment_name if none is given) to resume KD training (default=None)

              kd_model_external_checkpoint_path: The path to the external checkpoint to be loaded. Can be absolute or relative
                                               (ie: path/to/checkpoint.pth). If provided, will automatically attempt to
                                               load the checkpoint even if the load_checkpoint flag is not provided.
                                               (deafult=None)

        :keyword student_architecture: (Union[str, SgModule]) Defines the student's architecture from
            models/ALL_ARCHITECTURES (when str), or directly defined the student network (when SgModule).

        :keyword teacher_architecture: (Union[str, SgModule]) Defines the teacher's architecture from
            models/ALL_ARCHITECTURES (when str), or directly defined the teacher network (when SgModule).
            
        :keyword student_arch_params: (dict) Architecture H.P. e.g.: block, num_blocks, num_classes, etc for student
            net. (deafult={})

        :keyword teacher_arch_params: (dict) Architecture H.P. e.g.: block, num_blocks, num_classes, etc for teacher
            net. (deafult={})

        :keyword run_teacher_on_eval: (bool)- whether to run self.teacher at eval mode regardless of self.train(mode)


        """
        # TODO: VALIDATED INPUT ARGS
        # TODO: INSTANTIATE TEACHER AND STUDENT (AND VALIDATE TEACHER KNOWLEDGE)
        # TODO: CALL SUPER BUILD MODEL, AND OVERRIDE _LOAD_CHECKPOINT

        self._validate_args(arch_params, architecture, checkpoint_params, kwargs)

        student_architecture = get_param(kwargs, "student_architecture")
        teacher_architecture = get_param(kwargs, "teacher_architecture")
        student_arch_params = get_param(kwargs, "student_arch_params", default_val={})
        teacher_arch_params = get_param(kwargs, "teacher_arch_params", default_val={})
        run_teacher_on_eval = get_param(kwargs, "run_teacher_on_eval", default_val=False)

        student_pretrained_weights = get_param(checkpoint_params, 'student_pretrained_weights')
        teacher_pretrained_weights = get_param(checkpoint_params, 'teacher_pretrained_weights')

        student, _ = SgModel.instantiate_net(student_architecture, student_arch_params,
                                             {"pretrained_weights": student_pretrained_weights})

        teacher, _ = SgModel.instantiate_net(teacher_architecture, teacher_arch_params,
                                             {"pretrained_weights": teacher_pretrained_weights})

        super(KDModel, self).build_model(architecture=architecture, arch_params=arch_params,
                                         checkpoint_params=checkpoint_params, teacher=teacher, student=student, run_teacher_on_eval=run_teacher_on_eval)

        # student_architecture, _ = SgModel.instantiate_net(student_architecture, student_arch_params, {"pretrained_weights": get_param()})

        # # ASSIGN STUDENT'S NUM_CLASSES TO TEACHER AND MAIN KD MODULE ARCH PARAMS
        # teacher_arch_params['num_classes'] = student_arch_params['num_classes']
        # arch_params['num_classes'] = student_arch_params['num_classes']
        #
        # student_arch_params = core_utils.HpmStruct(**student_arch_params)
        # teacher_arch_params = core_utils.HpmStruct(**teacher_arch_params)
        #
        # if not student_net:
        #     student_pretrained_weights = core_utils.get_param(checkpoint_params, 'student_pretrained_weights',
        #                                                       default_val=None)
        #     student_net, _ = super_gradients.training.sg_model.sg_model.instantiate_net(student_architecture, student_arch_params,
        #                                                                                 student_pretrained_weights)
        #
        #
        #
        # if not teacher_net:
        #     teacher_net, _ = super_gradients.training.sg_model.sg_model.instantiate_net(teacher_architecture, teacher_arch_params,
        #                                                                                 teacher_pretrained_weights)

        # if teacher_checkpoint_path is not None:
        #
        #     #  WARN THAT TEACHER_CKPT WILL OVERRIDE TEACHER'S PRETRAINED WEIGHTS
        #     if teacher_pretrained_weights:
        #         logger.warning(
        #             teacher_checkpoint_path + " checkpoint is "
        #                                       "overriding " + teacher_pretrained_weights + " for teacher model")
        #
        #     # ALWAYS LOAD ITS EMA IF IT EXISTS
        #     load_teachers_ema = 'ema_net' in read_ckpt_state_dict(teacher_checkpoint_path).keys()
        #     load_checkpoint_to_model(ckpt_local_path=teacher_checkpoint_path,
        #                              load_backbone=False,
        #                              net=teacher_net,
        #                              strict='no_key_matching',
        #                              load_weights_only=True,
        #                              load_ema_as_net=load_teachers_ema)
        # # FIXME: MOVE TEACHER VALIDATION TO START
        #
        #
        #
        # arch_params['student'] = student_net
        # arch_params['teacher'] = teacher_net
        # # FIXME: OVERRIDE INSTANTIATE NET FOR KD MODEL
        # super(KDModel, self).build_model(architecture=architecture,
        #                                  arch_params=arch_params, checkpoint_params=checkpoint_params)

    def _validate_args(self, arch_params, architecture, checkpoint_params, kwargs):
        student_architecture = get_param(kwargs, "student_architecture")
        teacher_architecture = get_param(kwargs, "teacher_architecture")
        student_arch_params = get_param(kwargs, "student_arch_params", default_val={})
        teacher_arch_params = get_param(kwargs, "teacher_arch_params", default_val={})

        if get_param(checkpoint_params, 'pretrained_weights') is not None:
            raise ValueError("pretrained_weights is ambiguous for KD models.")

        if not isinstance(architecture, KDModule):
            if student_architecture is None or teacher_architecture is None:
                raise ValueError("When architecture is not intialized both student_architecture and "
                                 "teacher_architecture must be passed through **kwargs")
            if architecture not in KD_ARCHITECTURES.keys():
                raise TypeError("Unsupported KD architecture")

        # DERIVE NUMBER OF CLASSES FROM DATASET INTERFACE IF NOT SPECIFIED OR ARCH PARAMS FOR STUDENT
        if 'num_classes' not in student_arch_params.keys():
            if self.dataset_interface is None:
                raise Exception('Error', 'Number of classes not defined in students arch params and dataset is not '
                                         'defined')
            else:
                teacher_arch_params['num_classes'] = len(self.classes)

        # DERIVE NUMBER OF CLASSES FROM DATASET INTERFACE IF NOT SPECIFIED OR ARCH PARAMS FOR TEACHER
        if 'num_classes' not in teacher_arch_params.keys():
            if self.dataset_interface is None:
                raise Exception('Error', 'Number of classes not defined in teachers arch params and dataset is not '
                                         'defined')
            else:
                teacher_arch_params['num_classes'] = len(self.classes)

        if teacher_arch_params['num_classes'] != student_arch_params['num_classes']:
            raise ValueError("num_classes inconsistent between student and teacher architecture params")

        arch_params['num_classes'] = student_arch_params['num_classes']

        # MAKE SURE TEACHER'S PRETRAINED NUM CLASSES EQUALS TO THE ONES BELONGING TO STUDENT AS WE CAN'T REPLACE
        # THE TEACHER'S HEAD
        teacher_pretrained_weights = core_utils.get_param(checkpoint_params, 'teacher_pretrained_weights',
                                                          default_val=None)
        if teacher_pretrained_weights is not None:
            teacher_pretrained_num_classes = PRETRAINED_NUM_CLASSES[teacher_pretrained_weights]
            if teacher_pretrained_num_classes != teacher_arch_params['num_classes']:
                raise ValueError(
                    "Pretrained dataset number of classes in teacher's arch params must be equal to the student's "
                    "number of classes.")

        teacher_checkpoint_path = get_param(checkpoint_params, "teacher_checkpoint_path")

        # CHECK THAT TEACHER NETWORK HOLDS KNOWLEDGE FOR THE STUDENT TO LEARN FROM
        if not (teacher_pretrained_weights or teacher_checkpoint_path or checkpoint_params["load_checkpoint"]):
            raise ValueError("Expected: at least one of: teacher_pretrained_weights, teacher_checkpoint_path or "
                             "load_kd_model_checkpoint=True")

    @staticmethod
    def instantiate_net(architecture: Union[torch.nn.Module, SgModule.__class__, str], arch_params: dict,
                        checkpoint_params: dict, *args, **kwargs) -> tuple:

        student = get_param(kwargs, "student")
        teacher = get_param(kwargs, "teacher")
        run_teacher_on_eval = get_param(kwargs, "run_teacher_on_eval", default_val=False)

        architecture_cls = None
        if isinstance(architecture, str):
            architecture_cls = KD_ARCHITECTURES[architecture]
            net = architecture_cls(arch_params=arch_params, student=student, teacher=teacher, run_teacher_on_eval=run_teacher_on_eval)
        elif isinstance(architecture, KDModule.__class__):
            net = architecture(arch_params=arch_params, student=student, teacher=teacher, run_teacher_on_eval=run_teacher_on_eval)
        else:
            net = architecture

        return net, architecture_cls





