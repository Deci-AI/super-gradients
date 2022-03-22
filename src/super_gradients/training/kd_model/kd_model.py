from super_gradients.training.sg_model import SgModel
from typing import Union
from torch import nn
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import utils as core_utils
from super_gradients.training.utils import sg_model_utils
from super_gradients.training.utils.checkpoint_utils import read_ckpt_state_dict, load_checkpoint_to_model
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import get_param
from super_gradients.training.utils.callbacks import PhaseContext

logger = get_logger(__name__)


class KDModel(SgModel):
    """
    KDModel

    This class extends SgModel to support knowledge distillation.
    """

    def build_model(self,  # noqa: C901 - too complex
                    architecture: Union[str, nn.Module] = 'kd_module',
                    arch_params={},
                    *args, **kwargs):

        """
        :param architecture:               Should be ignored (currently, as we support only one kd_module)
         (defult='kd_module')

        :param arch_params:                A dictionary like object with the following keys/values:

          student: torch.nn.Module - the student model, if None is given- it will be instantiated according to
            student_architecture and student_arch_params (default=None)

          teacher: torch.nn.Module - the teacher model, if None is given- it will be instantiated according to
            teacher_architecture and teacher_arch_params (default=None)

          run_teacher_on_eval:   Whether to run self.teacher at eval mode regardless of self.train(mode) (default=False)

          student_architecture:       Defines the student's architecture from models/ALL_ARCHITECTURES (discarded in
           case 'student' is passed through arch_params) (default=none)

          teacher_architecture:       Defines the teacher's architecture from models/ALL_ARCHITECTURES (discarded in
           case 'teacher' is passed through arch_params) (default=None)

          student_arch_params:        Architecture H.P. e.g.: block, num_blocks, num_classes, etc for student net.
           (deafult={})

          teacher_arch_params:        Architecture H.P. e.g.: block, num_blocks, num_classes, etc for teacher net.
           (default={})

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
        """

        student_net = get_param(arch_params, "student")
        teacher_net = get_param(arch_params, "teacher")

        student_architecture = get_param(arch_params, "student_architecture")
        teacher_architecture = get_param(arch_params, "teacher_architecture")
        student_arch_params = get_param(arch_params, "student_arch_params")
        teacher_arch_params = get_param(arch_params, "teacher_arch_params")
        teacher_checkpoint_path = get_param(arch_params, "teacher_checkpoint_path")

        if (student_net and student_architecture) or (not student_net and not student_architecture):
            raise ValueError("Exactly one of: student, student_architecture should be passed through arch_params")
        if (teacher_net and teacher_architecture) or (not teacher_net and not teacher_architecture):
            raise ValueError("Exactly one of: teacher, teacher_architecture should be passed through arch_params")

        if not student_net:
            # DERIVE NUMBER OF CLASSES FROM DATASET INTERFACE IF NOT SPECIFIED
            if 'num_classes' not in student_arch_params.keys():
                if self.dataset_interface is None:
                    raise Exception('Error', 'Number of classes not defined in students arch params and dataset is not '
                                             'defined')
                else:
                    student_arch_params['num_classes'] = len(self.classes)

            # ASSIGN STUDENT'S NUM_CLASSES TO TEACHER AND MAIN KD MODULE ARCH PARAMS
            arch_params['num_classes'] = student_arch_params['num_classes']

            student_arch_params = core_utils.HpmStruct(**student_arch_params)
            student_net, _ = sg_model_utils.instantiate_net(student_architecture, student_arch_params)

        if not teacher_net:
            # MAKE SURE TEACHER'S PRETRAINED NUM CLASSES EQUALS TO THE ONES BELONGING TO STUDENT AS WE CAN'T REPLACE
            # THE TEACHER'S HEAD
            if 'num_classes' in teacher_arch_params and teacher_arch_params['num_classes'] != student_arch_params.num_classes:
                raise Exception('Error', f"Teacher's num_classes ({teacher_arch_params['num_classes']})"
                                         f" must match student num_classes ({student_arch_params.num_classes})")
            teacher_arch_params['num_classes'] = student_arch_params.num_classes
            teacher_arch_params = core_utils.HpmStruct(**teacher_arch_params)

            teacher_pretrained_weights = core_utils.get_param(teacher_arch_params, 'pretrained_weights', default_val=None)
            if teacher_pretrained_weights is not None:
                teacher_pretrained_num_classes = PRETRAINED_NUM_CLASSES[teacher_pretrained_weights]
                if teacher_pretrained_num_classes != arch_params['num_classes']:
                    raise ValueError(
                        "Pretrained dataset number of classes in teacher's arch params must be equal to the student's "
                        "number of classes.")

            teacher_net, _ = sg_model_utils.instantiate_net(teacher_architecture, teacher_arch_params)

            if teacher_checkpoint_path is not None:

                #  WARN THAT TEACHER_CKPT WILL OVERRIDE TEACHER'S PRETRAINED WEIGHTS
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

            # CHECK THAT TEACHER NETWORK HOLDS KNOWLEDGE FOR THE STUDENT TO LEARN FROM
            if not (teacher_pretrained_weights or teacher_checkpoint_path or arch_params["load_checkpoint"]):
                raise ValueError("Expected: at least one of: teacher_pretrained_weights, teacher_checkpoint_path or "
                                 "load_kd_model_checkpoint=True")

        arch_params['student'] = student_net
        arch_params['teacher'] = teacher_net
        arch_params["load_checkpoint"] = get_param(arch_params, "load_kd_model_checkpoint", False)
        arch_params["source_ckpt_folder_name"] = get_param(arch_params, "kd_model_source_ckpt_folder_name")
        arch_params["external_checkpoint_path"] = get_param(arch_params, "kd_model_external_checkpoint_path")

        super(KDModel, self).build_model(architecture=architecture,
                                         arch_params=arch_params)

    @staticmethod
    def update_context(context: PhaseContext, **kwargs):

        # OVERRIDE PREDS WITH STUDENT OUTPUT TO FIT METRIC CALLBACKS
        if 'preds' in kwargs.keys():
            kwargs['teacher_output'] = kwargs['preds'].teacher_output
            kwargs['preds'] = kwargs['preds'].student_output

        SgModel.update_context(context, **kwargs)
