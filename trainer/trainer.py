from omegaconf import DictConfig
import hydra
from super_gradients.training.sg_model import MultiGPUMode
from super_gradients.common.abstractions.abstract_logger import get_logger
import torch


class Trainer:
    """
    Class for running SuperGradient's recipes.
    See train_from_recipe example in the examples directory to demonstrate it's usage.
    """
    # FIXME: REMOVE PARAMETER MANIPULATION SPECIFIC FOR YOLO

    @staticmethod
    def scale_params_for_yolov5(cfg):
        """
        Scale:
            * learning rate,
            * weight decay,
            * box_loss_gain,
            * cls_loss_gain,
            * obj_loss_gain
        according to:
            * effective batch size
            * DDP world size
            * image size
            * num YOLO output layers
            * num classes
        """
        logger = get_logger(__name__)

        # Scale LR and weight decay
        is_ddp = cfg.sg_model.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_ddp else 1

        # Scale LR and WD for DDP due to gradients being averaged between devices
        # Equivalent to loss * WORLD_SIZE in ultralytics
        cfg.training_hyperparams.initial_lr *= world_size
        cfg.training_hyperparams.warmup_bias_lr *= world_size
        cfg.training_hyperparams.optimizer_params.weight_decay /= world_size

        # Scale WD with a factor of [effective batch size]/64.
        batch_size, batch_accumulate = cfg.dataset_params.batch_size, cfg.training_hyperparams.batch_accumulate
        batch_size_factor = cfg.sg_model.num_devices if is_ddp else cfg.sg_model.dataset_interface.batch_size_factor
        effective_batch_size = batch_size * batch_size_factor * batch_accumulate
        cfg.training_hyperparams.optimizer_params.weight_decay *= effective_batch_size / 64.

        # Scale EMA beta to match Ultralytics update
        cfg.training_hyperparams.ema_params.beta = cfg.training_hyperparams.max_epochs * len(cfg.sg_model.train_loader) / 2000.

        log_msg = \
            f"""
            IMPORTANT:\n
            Training with world size of {world_size}, {'DDP' if is_ddp else 'no DDP'}, effective batch size of {effective_batch_size},
            scaled:
                * initial_lr to {cfg.training_hyperparams.initial_lr};
                * warmup_bias_lr to {cfg.training_hyperparams.warmup_bias_lr};
                * weight_decay to {cfg.training_hyperparams.optimizer_params.weight_decay};
                * EMA beta to {cfg.training_hyperparams.ema_params.beta};
            """

        if cfg.training_hyperparams.loss == 'yolo_v5_loss':
            # Scale loss gains
            model = cfg.sg_model.net
            model = model.module if hasattr(model, 'module') else model
            num_levels = model._head._modules_list[-1].detection_layers_num
            train_image_size = cfg.dataset_params.train_image_size

            num_branches_norm = 3. / num_levels
            num_classes_norm = len(cfg.sg_model.classes) / 80.
            image_size_norm = train_image_size / 640.
            cfg.training_hyperparams.criterion_params.box_loss_gain *= num_branches_norm
            cfg.training_hyperparams.criterion_params.cls_loss_gain *= num_classes_norm * num_branches_norm
            cfg.training_hyperparams.criterion_params.obj_loss_gain *= image_size_norm ** 2 * num_branches_norm

            log_msg += \
                f"""
                * box_loss_gain to {cfg.training_hyperparams.criterion_params.box_loss_gain};
                * cls_loss_gain to {cfg.training_hyperparams.criterion_params.cls_loss_gain};
                * obj_loss_gain to {cfg.training_hyperparams.criterion_params.obj_loss_gain};
                """

        logger.info(log_msg)
        return cfg

    @staticmethod
    def train(cfg: DictConfig) -> None:
        """
        Trains according to cfg recipe configuration.

        @param cfg: The parsed DictConfig from yaml recipe files
        @return: output of sg_model.train(...) (i.e results tuple)
        """
        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        # CONNECT THE DATASET INTERFACE WITH DECI MODEL
        cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

        # BUILD NETWORK
        cfg.sg_model.build_model(cfg.architecture, arch_params=cfg.arch_params, load_checkpoint=cfg.load_checkpoint)

        # FIXME: REMOVE PARAMETER MANIPULATION SPECIFIC FOR YOLO
        if str(cfg.architecture).startswith("yolo_v5"):
            cfg = Trainer.scale_params_for_yolov5(cfg)

        # TRAIN
        cfg.sg_model.train(training_params=cfg.training_hyperparams)
