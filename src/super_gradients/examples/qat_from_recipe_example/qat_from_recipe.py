"""
Code for running PTQ/QAT on SuperGradients recipes.

THIS SCRIPT WILL MODIFY YOUR RECIPE TO SUIT QAT.
To use recipe as is, set `quantization_params.modify_recipe_for_qat.enable` to False

This script is proven NOT to work with DDP and will disable it automatically!

if `training_hyperparams.max_epochs=0`, only PTQ will be performed!

Usage:
    python qat_from_recipe.py
        --config-name=your_recipe
        +quantization_params=default_quantization_params OR your_desired_quantization_params
        +checkpoint_params.checkpoint_path=/full/path/to/your/checkpoint
"""

import os

import hydra
import pkg_resources
from omegaconf import DictConfig
from torch import nn

from super_gradients import Trainer, init_trainer
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.metrics.metric_utils import get_metrics_dict
from super_gradients.training.utils import get_param
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.training.utils.module_utils import fuse_repvgg_blocks_residual_branches
from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

logger = get_logger(__name__)


def quantize_and_calibrate(
    model: nn.Module,
    calibration_dataloader,
    num_calib_batches=16,
    method_w="max",
    method_i="mse",
    calibration_method=None,
    percentile=99.99,
    per_channel=True,
    learn_amax=False,
    skip_modules=None,
    verbose=False,
):
    model.eval()
    device = next(model.parameters()).device

    q_util = SelectiveQuantizer(
        default_quant_modules_calib_method_weights=method_w,
        default_quant_modules_calib_method_inputs=method_i,
        default_per_channel_quant_weights=per_channel,
        default_learn_amax=learn_amax,
    )

    if skip_modules is not None:
        q_util.register_skip_quantization(layer_names=set(skip_modules))

    calibrator = QuantizationCalibrator(verbose=verbose, torch_hist=True)

    # RepVGG and QARepVGG can be quantized only in the fused form
    fuse_repvgg_blocks_residual_branches(model)
    model.to(device=device)  # make sure that reparametrized modules are on the same device

    q_util.quantize_module(model)
    model.to(device=device)  # we have new modules, they can be on different devices

    calibration_method = calibration_method or method_i
    calibrator.calibrate_model(
        model,
        method=calibration_method,
        calib_data_loader=calibration_dataloader,
        num_calib_batches=num_calib_batches,
        percentile=percentile,
    )
    model.to(device=device)  # we can have _amax buffers scattered over different devices

    model.train()
    return model


def modify_training_params_for_qat(cfg):
    logger.info("Modifying recipe to suit QAT. Set quantization_params.modify_recipe_for_qat.enable=False to do it manually.")

    # Q/DQ Layers take a lot of space for activations in training mode
    if cfg.quantization_params.selective_quantizer_params.learn_amax:
        cfg.dataset_params.train_dataloader_params.batch_size //= cfg.quantization_params.modify_recipe_for_qat.batch_size_divisor
        cfg.dataset_params.val_dataloader_params.batch_size //= cfg.quantization_params.modify_recipe_for_qat.batch_size_divisor

        logger.warning(f"New dataset_params.train_dataloader_params.batch_size: {cfg.dataset_params.train_dataloader_params.batch_size}")
        logger.warning(f"New dataset_params.val_dataloader_params.batch_size: {cfg.dataset_params.val_dataloader_params.batch_size}")

    cfg.training_hyperparams.max_epochs //= cfg.quantization_params.modify_recipe_for_qat.max_epochs_divisor
    logger.warning(f"New number of epochs: {cfg.training_hyperparams.max_epochs}")

    lr_decay_factor = cfg.quantization_params.modify_recipe_for_qat.lr_decay_factor

    cfg.training_hyperparams.initial_lr *= lr_decay_factor
    if cfg.training_hyperparams.warmup_initial_lr is not None:
        cfg.training_hyperparams.warmup_initial_lr *= lr_decay_factor
    else:
        cfg.training_hyperparams.warmup_initial_lr = cfg.training_hyperparams.initial_lr * 0.01

    cfg.training_hyperparams.optimizer_params.weight_decay *= lr_decay_factor

    logger.warning(f"New learning rate: {cfg.training_hyperparams.initial_lr}")
    logger.warning(f"New weight decay: {cfg.training_hyperparams.optimizer_params.weight_decay}")

    # as recommended by pytorch-quantization docs
    cfg.training_hyperparams.lr_mode = "cosine"
    cfg.training_hyperparams.lr_warmup_epochs = (
        cfg.training_hyperparams.max_epochs // cfg.quantization_params.modify_recipe_for_qat.warmup_epochs_divisor
    ) or 1
    cfg.training_hyperparams.cosine_final_lr_ratio = cfg.quantization_params.modify_recipe_for_qat.cosine_final_lr_ratio

    # do mess with Q/DQ
    if cfg.training_hyperparams.ema:
        logger.warning("EMA will be disabled for QAT run.")
        cfg.training_hyperparams.ema = False

    if cfg.training_hyperparams.sync_bn:
        logger.warning("SyncBatchNorm will be disabled for QAT run.")
        cfg.training_hyperparams.sync_bn = False

    if cfg.quantization_params.modify_recipe_for_qat.disable_phase_callbacks and len(cfg.training_hyperparams.phase_callbacks) > 0:
        logger.warning(f"Recipe contains {len(cfg.training_hyperparams.phase_callbacks)} phase callbacks. All of them will be disabled.")
        cfg.training_hyperparams.phase_callbacks = []

    if cfg.multi_gpu != "OFF" or cfg.num_gpus != 1:
        logger.warning(f"Recipe requests multi_gpu={cfg.multi_gpu} and num_gpus={cfg.num_gpus}. Changing to multi_gpu=OFF and num_gpus=1")
        cfg.multi_gpu = "OFF"
        cfg.num_gpus = 1

    # no augmentations
    if "transforms" in cfg.dataset_params.val_dataset_params:
        cfg.dataset_params.train_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms


def qat_from_config(cfg):
    if "quantization_params" not in cfg:
        raise ValueError("Your recipe does not have quantization_params. Add them to use QAT.")

    if "checkpoint_path" not in cfg.checkpoint_params:
        raise ValueError("Starting checkpoint is a must for QAT finetuning.")

    if cfg.quantization_params.modify_recipe_for_qat.enable:
        modify_training_params_for_qat(cfg=cfg)

    setup_device(
        multi_gpu=core_utils.get_param(cfg, "multi_gpu", MultiGPUMode.OFF),
        num_gpus=core_utils.get_param(cfg, "num_gpus"),
        device="cuda",
    )

    train_dataloader = dataloaders.get(
        name=cfg.train_dataloader,
        dataset_params=cfg.dataset_params.train_dataset_params,
        dataloader_params=cfg.dataset_params.train_dataloader_params.copy(),
    )

    val_dataloader = dataloaders.get(
        name=cfg.val_dataloader,
        dataset_params=cfg.dataset_params.val_dataset_params,
        dataloader_params=cfg.dataset_params.val_dataloader_params.copy(),
    )

    # BUILD NETWORK
    model = models.get(
        model_name=cfg.arch_params.get("model_name", None) or cfg.architecture,
        num_classes=cfg.get("num_classes", None) or cfg.arch_params.num_classes,
        arch_params=cfg.arch_params,
        pretrained_weights=cfg.checkpoint_params.pretrained_weights,
        checkpoint_path=cfg.checkpoint_params.checkpoint_path,
        load_backbone=cfg.checkpoint_params.load_backbone,
    ).cuda()  # we assume that QAT is performed with CUDA

    quantize_and_calibrate(
        model,
        train_dataloader,
        method_w=cfg.quantization_params.selective_quantizer_params.method_w,
        method_i=cfg.quantization_params.selective_quantizer_params.method_i,
        per_channel=cfg.quantization_params.selective_quantizer_params.per_channel,
        learn_amax=cfg.quantization_params.selective_quantizer_params.learn_amax,
        skip_modules=cfg.quantization_params.selective_quantizer_params.skip_modules,
        num_calib_batches=cfg.quantization_params.calib_params.num_calib_batches or (512 // cfg.dataset_params.train_dataloader_params.batch_size) or 1,
        calibration_method=cfg.quantization_params.calib_params.calib_method,
        percentile=cfg.quantization_params.calib_params.percentile,
        verbose=cfg.quantization_params.calib_params.verbose,
    )

    logger.info("Performing validation of PTQ model...")

    trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=get_param(cfg, "ckpt_root_dir", default_val=None))
    val_results_tuple = trainer.test(model=model, test_loader=val_dataloader, test_metrics_list=cfg.training_hyperparams.valid_metrics_list)
    valid_metrics_dict = get_metrics_dict(val_results_tuple, trainer.test_metrics, trainer.loss_logging_items_names)
    results = ["Validate Results"]
    results += [f"   - {metric:10}: {value}" for metric, value in valid_metrics_dict.items()]
    logger.info("\n".join(results))

    if cfg.training_hyperparams.max_epochs != 0:
        # new Trainer object because calling Trainer.train after Trainer.test messes up init of the model
        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=get_param(cfg, "ckpt_root_dir", default_val=None))
        trainer.train(model=model, train_loader=train_dataloader, valid_loader=val_dataloader, training_params=cfg.training_hyperparams)
        suffix = "qat"
    else:
        logger.info("cfg.training_hyperparams.max_epochs is 0! Performing PTQ only!")
        suffix = "ptq"

    input_shape = next(iter(val_dataloader))[0].shape
    os.makedirs(trainer.checkpoints_dir_path, exist_ok=True)

    qdq_onnx_path = os.path.join(trainer.checkpoints_dir_path, f"{cfg.experiment_name}_{'x'.join((str(x) for x in input_shape))}_{suffix}.onnx")
    export_quantized_module_to_onnx(
        model=model.cpu(),
        onnx_filename=qdq_onnx_path,
        input_shape=input_shape,
        input_size=input_shape,
        train=False,
    )
    logger.info(f"Exporting {suffix.upper()} ONNX to {qdq_onnx_path}")


@hydra.main(config_path=pkg_resources.resource_filename("recipes", ""), version_base="1.2")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)

    return qat_from_config(cfg)


if __name__ == "__main__":
    init_trainer()
    main()
