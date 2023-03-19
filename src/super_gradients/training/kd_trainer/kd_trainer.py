from typing import Union, Dict, Mapping, Any

import hydra
import torch.nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from super_gradients.common import MultiGPUMode, StrictLoad
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import utils as core_utils, models
from super_gradients.training.dataloaders import dataloaders
from super_gradients.training.exceptions.kd_trainer_exceptions import (
    ArchitectureKwargsException,
    UnsupportedKDArchitectureException,
    InconsistentParamsException,
    UnsupportedKDModelArgException,
    TeacherKnowledgeException,
    UndefinedNumClassesException,
)
from super_gradients.training.models import SgModule
from super_gradients.common.registry.registry import KD_ARCHITECTURES
from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.utils import get_param, HpmStruct
from super_gradients.training.utils.callbacks import KDModelMetricsUpdateCallback
from super_gradients.training.utils.checkpoint_utils import read_ckpt_state_dict, load_checkpoint_to_model
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.utils.ema import KDModelEMA

logger = get_logger(__name__)


class KDTrainer(Trainer):
    def __init__(self, experiment_name: str, device: str = None, multi_gpu: Union[MultiGPUMode, str] = None, ckpt_root_dir: str = None):
        super().__init__(experiment_name=experiment_name, device=device, multi_gpu=multi_gpu, ckpt_root_dir=ckpt_root_dir)
        self.student_architecture = None
        self.teacher_architecture = None
        self.student_arch_params = None
        self.teacher_arch_params = None

    @classmethod
    def train_from_config(cls, cfg: Union[DictConfig, dict]) -> None:
        """
        Trains according to cfg recipe configuration.

        :param cfg: The parsed DictConfig from yaml recipe files
        :return: output of kd_trainer.train(...) (i.e results tuple)
        """
        # INSTANTIATE ALL OBJECTS IN CFG
        setup_device(
            device=core_utils.get_param(cfg, "device"),
            multi_gpu=core_utils.get_param(cfg, "multi_gpu"),
            num_gpus=core_utils.get_param(cfg, "num_gpus"),
        )

        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        trainer = KDTrainer(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)

        # INSTANTIATE DATA LOADERS
        train_dataloader = dataloaders.get(
            name=cfg.train_dataloader, dataset_params=cfg.dataset_params.train_dataset_params, dataloader_params=cfg.dataset_params.train_dataloader_params
        )

        val_dataloader = dataloaders.get(
            name=cfg.val_dataloader, dataset_params=cfg.dataset_params.val_dataset_params, dataloader_params=cfg.dataset_params.val_dataloader_params
        )

        student = models.get(
            cfg.student_architecture,
            arch_params=cfg.student_arch_params,
            strict_load=cfg.student_checkpoint_params.strict_load,
            pretrained_weights=cfg.student_checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.student_checkpoint_params.checkpoint_path,
            load_backbone=cfg.student_checkpoint_params.load_backbone,
        )

        teacher = models.get(
            cfg.teacher_architecture,
            arch_params=cfg.teacher_arch_params,
            strict_load=cfg.teacher_checkpoint_params.strict_load,
            pretrained_weights=cfg.teacher_checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.teacher_checkpoint_params.checkpoint_path,
            load_backbone=cfg.teacher_checkpoint_params.load_backbone,
        )

        recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}

        # TRAIN
        trainer.train(
            training_params=cfg.training_hyperparams,
            student=student,
            teacher=teacher,
            kd_architecture=cfg.architecture,
            kd_arch_params=cfg.arch_params,
            run_teacher_on_eval=cfg.run_teacher_on_eval,
            train_loader=train_dataloader,
            valid_loader=val_dataloader,
            additional_configs_to_log=recipe_logged_cfg,
        )

    def _validate_args(self, arch_params, architecture, checkpoint_params, **kwargs):
        student_architecture = get_param(kwargs, "student_architecture")
        teacher_architecture = get_param(kwargs, "teacher_architecture")
        student_arch_params = get_param(kwargs, "student_arch_params")
        teacher_arch_params = get_param(kwargs, "teacher_arch_params")

        if get_param(checkpoint_params, "pretrained_weights") is not None:
            raise UnsupportedKDModelArgException("pretrained_weights", "checkpoint_params")

        if not isinstance(architecture, KDModule):
            if student_architecture is None or teacher_architecture is None:
                raise ArchitectureKwargsException()
            if architecture not in KD_ARCHITECTURES.keys():
                raise UnsupportedKDArchitectureException(architecture)

        # DERIVE NUMBER OF CLASSES FROM DATASET INTERFACE IF NOT SPECIFIED OR ARCH PARAMS FOR TEACHER AND STUDENT
        self._validate_num_classes(student_arch_params, teacher_arch_params)

        arch_params["num_classes"] = student_arch_params["num_classes"]

        # MAKE SURE TEACHER'S PRETRAINED NUM CLASSES EQUALS TO THE ONES BELONGING TO STUDENT AS WE CAN'T REPLACE
        # THE TEACHER'S HEAD
        teacher_pretrained_weights = core_utils.get_param(checkpoint_params, "teacher_pretrained_weights", default_val=None)
        if teacher_pretrained_weights is not None:
            teacher_pretrained_num_classes = PRETRAINED_NUM_CLASSES[teacher_pretrained_weights]
            if teacher_pretrained_num_classes != teacher_arch_params["num_classes"]:
                raise InconsistentParamsException(
                    "Pretrained dataset number of classes", "teacher's arch params", "number of classes", "student's number of classes"
                )

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
        if teacher_arch_params["num_classes"] != student_arch_params["num_classes"]:
            raise InconsistentParamsException("num_classes", "student_arch_params", "num_classes", "teacher_arch_params")

    def _validate_subnet_num_classes(self, subnet_arch_params):
        """
        Derives num_classes in student_arch_params/teacher_arch_params from dataset interface or raises an error
         when none is given

        :param subnet_arch_params: Arch params for student/teacher

        """

        if "num_classes" not in subnet_arch_params.keys():
            if self.dataset_interface is None:
                raise UndefinedNumClassesException()
            else:
                subnet_arch_params["num_classes"] = len(self.classes)

    def _instantiate_net(self, architecture: Union[KDModule, KDModule.__class__, str], arch_params: dict, checkpoint_params: dict, *args, **kwargs) -> tuple:
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
        student_pretrained_weights = get_param(checkpoint_params, "student_pretrained_weights")
        teacher_pretrained_weights = get_param(checkpoint_params, "teacher_pretrained_weights")

        student = super()._instantiate_net(student_architecture, student_arch_params, {"pretrained_weights": student_pretrained_weights})
        teacher = super()._instantiate_net(teacher_architecture, teacher_arch_params, {"pretrained_weights": teacher_pretrained_weights})

        run_teacher_on_eval = get_param(kwargs, "run_teacher_on_eval", default_val=False)

        return self._instantiate_kd_net(arch_params, architecture, run_teacher_on_eval, student, teacher)

    def _instantiate_kd_net(self, arch_params, architecture, run_teacher_on_eval, student, teacher):
        if isinstance(architecture, str):
            architecture_cls = KD_ARCHITECTURES[architecture]
            net = architecture_cls(arch_params=arch_params, student=student, teacher=teacher, run_teacher_on_eval=run_teacher_on_eval)
        elif isinstance(architecture, KDModule.__class__):
            net = architecture(arch_params=arch_params, student=student, teacher=teacher, run_teacher_on_eval=run_teacher_on_eval)
        else:
            net = architecture
        return net

    def _load_checkpoint_to_model(self):
        """
        Initializes teacher weights with teacher_checkpoint_path if needed, then handles checkpoint loading for
         the entire KD network following the same logic as in Trainer.
        """
        teacher_checkpoint_path = get_param(self.checkpoint_params, "teacher_checkpoint_path")
        teacher_net = self.net.module.teacher

        if teacher_checkpoint_path is not None:

            #  WARN THAT TEACHER_CKPT WILL OVERRIDE TEACHER'S PRETRAINED WEIGHTS
            teacher_pretrained_weights = get_param(self.checkpoint_params, "teacher_pretrained_weights")
            if teacher_pretrained_weights:
                logger.warning(teacher_checkpoint_path + " checkpoint is " "overriding " + teacher_pretrained_weights + " for teacher model")

            # ALWAYS LOAD ITS EMA IF IT EXISTS
            load_teachers_ema = "ema_net" in read_ckpt_state_dict(teacher_checkpoint_path).keys()
            load_checkpoint_to_model(
                ckpt_local_path=teacher_checkpoint_path,
                load_backbone=False,
                net=teacher_net,
                strict=StrictLoad.NO_KEY_MATCHING,
                load_weights_only=True,
                load_ema_as_net=load_teachers_ema,
            )

        super(KDTrainer, self)._load_checkpoint_to_model()

    def _add_metrics_update_callback(self, phase):
        """
        Adds KDModelMetricsUpdateCallback to be fired at phase

        :param phase: Phase for the metrics callback to be fired at
        """
        self.phase_callbacks.append(KDModelMetricsUpdateCallback(phase))

    def _get_hyper_param_config(self):
        """
        Creates a training hyper param config for logging with additional KD related hyper params.
        """
        hyper_param_config = super()._get_hyper_param_config()
        hyper_param_config.update(
            {
                "student_architecture": self.student_architecture,
                "teacher_architecture": self.teacher_architecture,
                "student_arch_params": self.student_arch_params,
                "teacher_arch_params": self.teacher_arch_params,
            }
        )
        return hyper_param_config

    def _instantiate_ema_model(self, ema_params: Mapping[str, Any]) -> KDModelEMA:
        """Instantiate ema model for standard SgModule.
        :param decay_type: (str) The decay climb schedule. See EMA_DECAY_FUNCTIONS for more details.
        :param decay: The maximum decay value. As the training process advances, the decay will climb towards this value
                      according to decay_type schedule. See EMA_DECAY_FUNCTIONS for more details.
        :param kwargs: Additional parameters for the decay function. See EMA_DECAY_FUNCTIONS for more details.
        """
        logger.info(f"Using EMA with params {ema_params}")
        return KDModelEMA.from_params(self.net, **ema_params)

    def _save_best_checkpoint(self, epoch, state):
        """
        Overrides parent best_ckpt saving to modify the state dict so that we only save the student.
        """
        if self.ema:
            best_net = core_utils.WrappedModel(self.ema_model.ema.module.student)
            state.pop("ema_net")
        else:
            best_net = core_utils.WrappedModel(self.net.module.student)

        state["net"] = best_net.state_dict()
        self.sg_logger.add_checkpoint(tag=self.ckpt_best_name, state_dict=state, global_step=epoch)

    def train(
        self,
        model: KDModule = None,
        training_params: Dict = None,
        student: SgModule = None,
        teacher: torch.nn.Module = None,
        kd_architecture: Union[KDModule.__class__, str] = "kd_module",
        kd_arch_params: Dict = None,
        run_teacher_on_eval=False,
        train_loader: DataLoader = None,
        valid_loader: DataLoader = None,
        additional_configs_to_log: Dict = None,
        *args,
        **kwargs,
    ):
        """
        Trains the student network (wrapped in KDModule network).


        :param model: KDModule, network to train. When none is given will initialize KDModule according to kd_architecture,
            student and teacher (default=None)
        :param training_params: dict, Same as in Trainer.train()
        :param student: SgModule - the student trainer
        :param teacher: torch.nn.Module- the teacher trainer
        :param kd_architecture: KDModule architecture to use, currently only 'kd_module' is supported (default='kd_module').
        :param kd_arch_params: architecture params to pas to kd_architecture constructor.
        :param run_teacher_on_eval: bool- whether to run self.teacher at eval mode regardless of self.train(mode)
        :param train_loader: Dataloader for train set.
        :param valid_loader: Dataloader for validation.
        :param additional_configs_to_log: Dict, dictionary containing configs that will be added to the training's
                sg_logger. Format should be {"Config_title_1": {...}, "Config_title_2":{..}}, (optional, default=None)
        """
        kd_net = self.net or model
        kd_arch_params = kd_arch_params or dict()
        if kd_net is None:
            if student is None or teacher is None:
                raise ValueError("Must pass student and teacher models or net (KDModule).")
            kd_net = self._instantiate_kd_net(
                arch_params=HpmStruct(**kd_arch_params), architecture=kd_architecture, run_teacher_on_eval=run_teacher_on_eval, student=student, teacher=teacher
            )
        super(KDTrainer, self).train(
            model=kd_net,
            training_params=training_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
            additional_configs_to_log=additional_configs_to_log,
        )
