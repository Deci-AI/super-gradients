from typing import Union
import torch

from super_gradients.training.exceptions.kd_model_exceptions import UnsupportedKDModelArgException, \
    ArchitectureKwargsException, UnsupportedKDArchitectureException, InconsistentParamsException, \
    TeacherKnowledgeException, UndefinedNumClassesException
from super_gradients.training.models.all_architectures import ARCHITECTURES, KD_ARCHITECTURES
from super_gradients.training import utils as core_utils
from super_gradients.training.models import SgModule
from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import get_param
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model, load_pretrained_weights, \
    read_ckpt_state_dict
import torch.nn as nn


class SgNetsFactory:
    @classmethod
    def get(cls, architecture: Union[str, nn.Module], arch_params={}, checkpoint_params={}, *args,
            **kwargs) -> nn.Module:
        """
        :param architecture:               Defines the network's architecture from models/ALL_ARCHITECTURES
        :param arch_params:                Architecture H.P. e.g.: block, num_blocks, num_classes, etc.
        :param checkpoint_params:          Dictionary like object with the following key:values:

            strict_load:                See StrictLoad class documentation for details.
            load_backbone:              loads the provided checkpoint to net.backbone instead of net
            external_checkpoint_path:   The path to the external checkpoint to be loaded. Can be absolute or relative
                                               (ie: path/to/checkpoint.pth). If provided, will automatically attempt to
                                               load the checkpoint even if the load_checkpoint flag is not provided.

        """

        arch_params = core_utils.HpmStruct(**arch_params)
        checkpoint_params = core_utils.HpmStruct(**checkpoint_params)
        net = cls.instantiate_net(architecture, arch_params, checkpoint_params, *args, **kwargs)
        strict_load = core_utils.get_param(checkpoint_params, 'strict_load', default_val="no_key_matching")
        load_backbone = core_utils.get_param(checkpoint_params, 'load_backbone', default_val=False)
        checkpoint_path = core_utils.get_param(checkpoint_params, 'checkpoint_path')
        if checkpoint_path:
            load_ema_as_net = 'ema_net' in read_ckpt_state_dict(ckpt_path=checkpoint_path).keys()
            _ = load_checkpoint_to_model(ckpt_local_path=checkpoint_path,
                                         load_backbone=load_backbone,
                                         net=net,
                                         strict=strict_load.value if hasattr(strict_load, "value") else strict_load,
                                         load_weights_only=True,
                                         load_ema_as_net=load_ema_as_net)
        return net

    @classmethod
    def instantiate_net(cls, architecture: Union[torch.nn.Module, SgModule.__class__, str], arch_params: dict,
                        checkpoint_params: dict, *args, **kwargs) -> nn.Module:
        """
        Instantiates nn.Module according to architecture and arch_params, and handles pretrained weights and the required
            module manipulation (i.e head replacement).

        :param architecture: String, torch.nn.Module or uninstantiated SgModule class describing the netowrks architecture.
        :param arch_params: Architecture's parameters passed to networks c'tor.
        :param checkpoint_params: checkpoint loading related parameters dictionary with 'pretrained_weights' key,
            s.t it's value is a string describing the dataset of the pretrained weights (for example "imagenent").

        :return: instantiated netowrk i.e torch.nn.Module, architecture_class (will be none when architecture is not str)

        """
        pretrained_weights = core_utils.get_param(checkpoint_params, 'pretrained_weights', default_val=None)

        if pretrained_weights is not None:
            if hasattr(arch_params, "num_classes"):
                num_classes_new_head = arch_params.num_classes
            else:
                num_classes_new_head = PRETRAINED_NUM_CLASSES[pretrained_weights]

            arch_params.num_classes = PRETRAINED_NUM_CLASSES[pretrained_weights]

        if isinstance(architecture, str):
            architecture_cls = ARCHITECTURES[architecture]
            net = architecture_cls(arch_params=arch_params)
        elif isinstance(architecture, SgModule.__class__):
            net = architecture(arch_params)
        else:
            net = architecture

        if pretrained_weights:
            load_pretrained_weights(net, architecture, pretrained_weights)
            if num_classes_new_head != arch_params.num_classes:
                net.replace_head(new_num_classes=num_classes_new_head)
                arch_params.num_classes = num_classes_new_head

        return net

class KDNetsFactory(SgNetsFactory):
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
        if not (teacher_pretrained_weights or teacher_checkpoint_path or load_kd_model_checkpoint or isinstance(teacher_architecture, nn.Module)):
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

def get(architecture: Union[str, nn.Module], arch_params={}, checkpoint_params={}, *args, **kwargs) -> nn.Module:
    return SgNetsFactory.get(architecture, arch_params, checkpoint_params, *args, **kwargs)

