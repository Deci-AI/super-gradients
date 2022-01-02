# Yolo v5 Detection training on CoCo2017 Dataset:
# Yolo v5s train on 320x320 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~28.4
# Yolo v5s train in 640x640 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~29.1

# Yolo v5 Detection training on CoCo2014 Dataset:
# Yolo v5s train on 320x320 mAP@0.5-0.95 (confidence 0.001, test on 320x320 images) ~28.77

# batch size may need to change depending on model size and GPU (2080Ti, V100)
# The code is optimized for running with a Mini-Batch of 64 examples... So depending on the amount of GPUs,
# you should change the "batch_accumulate" param in the training_params dict to be batch_size * gpu_num * batch_accumulate = 64.

import pkg_resources

import torch
import hydra
from omegaconf import DictConfig

import super_gradients
from super_gradients.training.sg_model import MultiGPUMode
from super_gradients.common.abstractions.abstract_logger import get_logger


def scale_params(cfg):
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
    cfg.training_params.initial_lr *= world_size
    cfg.training_params.warmup_bias_lr *= world_size
    cfg.training_params.optimizer_params.weight_decay /= world_size

    # Scale WD with a factor of [effective batch size]/64.
    batch_size, batch_accumulate = cfg.dataset_params.batch_size, cfg.training_params.batch_accumulate
    batch_size_factor = cfg.sg_model.num_devices if is_ddp else cfg.sg_model.dataset_interface.batch_size_factor
    effective_batch_size = batch_size * batch_size_factor * batch_accumulate
    cfg.training_params.optimizer_params.weight_decay *= effective_batch_size / 64.

    # Scale EMA beta to match Ultralytics update
    cfg.training_params.ema_params.beta = cfg.training_params.max_epochs * len(cfg.sg_model.train_loader) / 2000.

    log_msg = \
        f"""

        IMPORTANT:\n
        Training with world size of {world_size}, {'DDP' if is_ddp else 'no DDP'}, effective batch size of {effective_batch_size},
        scaled:
            * initial_lr to {cfg.training_params.initial_lr};
            * warmup_bias_lr to {cfg.training_params.warmup_bias_lr};
            * weight_decay to {cfg.training_params.optimizer_params.weight_decay};
            * EMA beta to {cfg.training_params.ema_params.beta};

        """

    if cfg.training_params.loss == 'yolo_v5_loss':
        # Scale loss gains
        model = cfg.sg_model.net
        model = model.module if hasattr(model, 'module') else model
        num_levels = model._head._modules_list[-1].detection_layers_num
        train_image_size = cfg.dataset_params.train_image_size

        num_branches_norm = 3. / num_levels
        num_classes_norm = len(cfg.sg_model.classes) / 80.
        image_size_norm = train_image_size / 640.
        cfg.training_params.criterion_params.box_loss_gain *= num_branches_norm
        cfg.training_params.criterion_params.cls_loss_gain *= num_classes_norm * num_branches_norm
        cfg.training_params.criterion_params.obj_loss_gain *= image_size_norm ** 2 * num_branches_norm

        log_msg += \
            f"""
            * box_loss_gain to {cfg.training_params.criterion_params.box_loss_gain};
            * cls_loss_gain to {cfg.training_params.criterion_params.cls_loss_gain};
            * obj_loss_gain to {cfg.training_params.criterion_params.obj_loss_gain};

            """

    logger.info(log_msg)
    return cfg


@hydra.main(config_path=pkg_resources.resource_filename("conf", ""), config_name="coco2017_yolov5_conf")
def train(cfg: DictConfig) -> None:
    # INSTANTIATE ALL OBJECTS IN CFG
    cfg = hydra.utils.instantiate(cfg)

    # CONNECT THE DATASET INTERFACE WITH DECI MODEL
    cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

    # BUILD NETWORK
    cfg.sg_model.build_model(cfg.architecture, load_checkpoint=cfg.load_checkpoint)

    cfg = scale_params(cfg)

    # TRAIN
    cfg.sg_model.train(training_params=cfg.training_params)


if __name__ == "__main__":
    super_gradients.init_trainer()
    train()
