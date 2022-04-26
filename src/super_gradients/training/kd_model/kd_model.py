import torch.nn

from super_gradients.training.models.all_architectures import KD_ARCHITECTURES
from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.sg_model import SgModel
from typing import Union
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import utils as core_utils
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import get_param
from super_gradients.training.utils.checkpoint_utils import read_ckpt_state_dict, \
    load_checkpoint_to_model
from super_gradients.training.exceptions.kd_model_exceptions import ArchitectureKwargsException, \
    UnsupportedKDArchitectureException, InconsistentParamsException, UnsupportedKDModelArgException, \
    TeacherKnowledgeException, UndefinedNumClassesException
from super_gradients.training.utils.callbacks import KDModelMetricsUpdateCallback
logger = get_logger(__name__)


class KDModel(SgModel):
    def __init__(self, *args, **kwargs):
        super(KDModel, self).__init__(*args, **kwargs)
        self.student_architecture = None
        self.teacher_architecture = None
        self.student_arch_params = None
        self.teacher_arch_params = None

    def build_model(self,
                    # noqa: C901 - too complex
                    architecture: Union[str, KDModule] = 'kd_module',
                    arch_params={}, checkpoint_params={},
                    *args, **kwargs):
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
        kwargs.setdefault("student_architecture", None)
        kwargs.setdefault("teacher_architecture", None)
        kwargs.setdefault("student_arch_params", {})
        kwargs.setdefault("teacher_arch_params", {})
        kwargs.setdefault("run_teacher_on_eval", False)

        self._validate_args(arch_params, architecture, checkpoint_params, **kwargs)

        self.student_architecture = kwargs.get("student_architecture")
        self.teacher_architecture = kwargs.get("teacher_architecture")
        self.student_arch_params = kwargs.get("student_arch_params")
        self.teacher_arch_params = kwargs.get("teacher_arch_params")

        super(KDModel, self).build_model(architecture=architecture, arch_params=arch_params,
                                         checkpoint_params=checkpoint_params, **kwargs)

    def _validate_args(self, arch_params, architecture, checkpoint_params, **kwargs):
        student_architecture = get_param(kwargs, "student_architecture")
        teacher_architecture = get_param(kwargs, "teacher_architecture")
        student_arch_params = get_param(kwargs, "student_arch_params")
        teacher_arch_params = get_param(kwargs, "teacher_arch_params")

        if get_param(checkpoint_params, 'pretrained_weights') is not None:
            raise UnsupportedKDModelArgException("pretrained_weights", "checkpoint_params")

        if not isinstance(architecture, KDModule):
            if student_architecture is None or teacher_architecture is None:
                raise ArchitectureKwargsException()
            if architecture not in KD_ARCHITECTURES.keys():
                raise UnsupportedKDArchitectureException(architecture)

        # DERIVE NUMBER OF CLASSES FROM DATASET INTERFACE IF NOT SPECIFIED OR ARCH PARAMS FOR TEACHER AND STUDENT
        self._validate_num_classes(student_arch_params, teacher_arch_params)

        arch_params['num_classes'] = student_arch_params['num_classes']

        # MAKE SURE TEACHER'S PRETRAINED NUM CLASSES EQUALS TO THE ONES BELONGING TO STUDENT AS WE CAN'T REPLACE
        # THE TEACHER'S HEAD
        teacher_pretrained_weights = core_utils.get_param(checkpoint_params, 'teacher_pretrained_weights',
                                                          default_val=None)
        if teacher_pretrained_weights is not None:
            teacher_pretrained_num_classes = PRETRAINED_NUM_CLASSES[teacher_pretrained_weights]
            if teacher_pretrained_num_classes != teacher_arch_params['num_classes']:
                raise InconsistentParamsException("Pretrained dataset number of classes", "teacher's arch params",
                                                  "number of classes", "student's number of classes")

        teacher_checkpoint_path = get_param(checkpoint_params, "teacher_checkpoint_path")
        load_kd_model_checkpoint = get_param(checkpoint_params, "load_checkpoint")

        # CHECK THAT TEACHER NETWORK HOLDS KNOWLEDGE FOR THE STUDENT TO LEARN FROM OR THAT WE ARE LOADING AN ENTIRE KD
        if not (teacher_pretrained_weights or teacher_checkpoint_path or load_kd_model_checkpoint or isinstance(teacher_architecture, torch.nn.Module)):
            raise TeacherKnowledgeException()

    def _validate_num_classes(self, student_arch_params, teacher_arch_params):
        """
        Checks validity of num_classes for num_classes (i.e existence and consistency between subnets)

        :param student_arch_params: (dict) Architecture H.P. e.g.: block, num_blocks, num_classes, etc for student
        :param teacher_arch_params: (dict) Architecture H.P. e.g.: block, num_blocks, num_classes, etc for teacher

        """
        self._validate_subnet_num_classes(student_arch_params)
        self._validate_subnet_num_classes(teacher_arch_params)
        if teacher_arch_params['num_classes'] != student_arch_params['num_classes']:
            raise InconsistentParamsException("num_classes", "student_arch_params", "num_classes",
                                              "teacher_arch_params")

    def _validate_subnet_num_classes(self, subnet_arch_params):
        """
        Derives num_classes in student_arch_params/teacher_arch_params from dataset interface or raises an error
         when none is given

        :param subnet_arch_params: Arch params for student/teacher

        """

        if 'num_classes' not in subnet_arch_params.keys():
            if self.dataset_interface is None:
                raise UndefinedNumClassesException()
            else:
                subnet_arch_params['num_classes'] = len(self.classes)

    def instantiate_net(self, architecture: Union[KDModule, KDModule.__class__, str], arch_params: dict,
                        checkpoint_params: dict, *args, **kwargs) -> tuple:
        """
        Instantiates kd_module according to architecture and arch_params, handles pretrained weights for the student
         and teacher networks, and the required module manipulation (i.e head replacement) for the teacher network.

        :param architecture: String, KDModule or uninstantiated KDModule class describing the netowrks architecture.
        :param arch_params: Architecture's parameters passed to networks c'tor.
        :param checkpoint_params: checkpoint loading related parameters dictionary with 'pretrained_weights' key,
            s.t it's value is a string describing the dataset of the pretrained weights (for example "imagenent").

        :return: instantiated netowrk i.e KDModule, architecture_class (will be none when architecture is not str)
        """

        student_architecture = get_param(kwargs, "student_architecture")
        teacher_architecture = get_param(kwargs, "teacher_architecture")
        student_arch_params = get_param(kwargs, "student_arch_params")
        teacher_arch_params = get_param(kwargs, "teacher_arch_params")
        student_arch_params = core_utils.HpmStruct(**student_arch_params)
        teacher_arch_params = core_utils.HpmStruct(**teacher_arch_params)
        student_pretrained_weights = get_param(checkpoint_params, 'student_pretrained_weights')
        teacher_pretrained_weights = get_param(checkpoint_params, 'teacher_pretrained_weights')

        student = super().instantiate_net(student_architecture, student_arch_params,
                                          {"pretrained_weights": student_pretrained_weights})
        teacher = super().instantiate_net(teacher_architecture, teacher_arch_params,
                                          {"pretrained_weights": teacher_pretrained_weights})

        run_teacher_on_eval = get_param(kwargs, "run_teacher_on_eval", default_val=False)

        if isinstance(architecture, str):
            architecture_cls = KD_ARCHITECTURES[architecture]
            net = architecture_cls(arch_params=arch_params, student=student, teacher=teacher,
                                   run_teacher_on_eval=run_teacher_on_eval)
        elif isinstance(architecture, KDModule.__class__):
            net = architecture(arch_params=arch_params, student=student, teacher=teacher,
                               run_teacher_on_eval=run_teacher_on_eval)
        else:
            net = architecture

        return net

    def _load_checkpoint_to_model(self):
        """
        Initializes teacher weights with teacher_checkpoint_path if needed, then handles checkpoint loading for
         the entire KD network following the same logic as in SgModel.
        """
        teacher_checkpoint_path = get_param(self.checkpoint_params, "teacher_checkpoint_path")
        teacher_net = self.net.module.teacher

        if teacher_checkpoint_path is not None:

            #  WARN THAT TEACHER_CKPT WILL OVERRIDE TEACHER'S PRETRAINED WEIGHTS
            teacher_pretrained_weights = get_param(self.checkpoint_params, "teacher_pretrained_weights")
            if teacher_pretrained_weights:
                logger.warning(
                    teacher_checkpoint_path + " checkpoint is "
                                              "overriding " + teacher_pretrained_weights + " for teacher model")

            # ALWAYS LOAD ITS EMA IF IT EXISTS
            load_teachers_ema = 'ema_net' in read_ckpt_state_dict(teacher_checkpoint_path).keys()
            load_checkpoint_to_model(ckpt_local_path=teacher_checkpoint_path,
                                     load_backbone=False,
                                     net=teacher_net,
                                     strict='no_key_matching',
                                     load_weights_only=True,
                                     load_ema_as_net=load_teachers_ema)

        super(KDModel, self)._load_checkpoint_to_model()

    def _add_metrics_update_callback(self, phase):
        """
        Adds KDModelMetricsUpdateCallback to be fired at phase

        :param phase: Phase for the metrics callback to be fired at
        """
        self.phase_callbacks.append(KDModelMetricsUpdateCallback(phase))

    def get_hyper_param_config(self):
        """
        Creates a training hyper param config for logging with additional KD related hyper params.
        """
        hyper_param_config = super().get_hyper_param_config()
        hyper_param_config.update({"student_architecture": self.student_architecture,
                                   "teacher_architecture": self.teacher_architecture,
                                   "student_arch_params": self.student_arch_params,
                                   "teacher_arch_params": self.teacher_arch_params
                                   })
        return hyper_param_config
