from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch import nn

from super_gradients import Trainer, init_trainer
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.data_types.enum import MultiGPUMode
from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.utils.checkpoint_utils import get_checkpoints_dir_path
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
    calibrator.calibrate_model(model, method=calibration_method, calib_data_loader=calibration_dataloader, num_calib_batches=num_calib_batches)
    model.to(device=device)  # we can have _amax buffers scattered over different devices

    model.train()
    return model


def modify_training_params_for_qat(cfg):
    # Q/DQ Layers take a lot of space for activations in training mode
    if cfg.qat_params.sq_params.get("learn_amax", False):
        cfg.dataset_params.train_dataloader_params.batch_size //= 2
        cfg.dataset_params.val_dataloader_params.batch_size //= 2

    # 10% of the training regime
    cfg.training_hyperparams.max_epochs //= 10

    # very small initial LR and WD
    lr_decay_factor = cfg.qat_params.get("lr_decay_factor", 100.0)
    cfg.training_hyperparams.initial_lr /= lr_decay_factor
    if cfg.training_hyperparams.warmup_initial_lr is not None:
        cfg.training_hyperparams.warmup_initial_lr /= lr_decay_factor
    else:
        cfg.training_hyperparams.warmup_initial_lr = cfg.training_hyperparams.initial_lr / 100.0

    cfg.training_hyperparams.optimizer_params.weight_decay /= lr_decay_factor

    # as recommended by pytorch-quantization docs
    cfg.training_hyperparams.lr_mode = "cosine"
    cfg.training_hyperparams.lr_warmup_epochs = (cfg.training_hyperparams.max_epochs // 10) or 1
    cfg.training_hyperparams.cosine_final_lr_ratio = 0.01

    # do mess with Q/DQ
    cfg.training_hyperparams.ema = False
    cfg.training_hyperparams.sync_bn = False
    cfg.training_hyperparams.phase_callbacks = []
    cfg.multi_gpu = "OFF"
    cfg.num_gpus = 1

    # no augmentations
    cfg.dataset_params.train_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms


@hydra.main(config_path="recipes", version_base="1.2")
def main(cfg: DictConfig) -> None:
    if "qat_params" not in cfg:
        raise ValueError("Your recipe does not have qat_params. Add them to use QAT.")

    if "checkpoint_path" not in cfg.checkpoint_params:
        raise ValueError("Starting checkpoint is a must for QAT finetuning.")

    if cfg.qat_params.modify_recipe_params:
        modify_training_params_for_qat(cfg=cfg)
        logger.info("Modifying recipe to suit QAT. Add qat_params.modify_recipe_params=False to do it manually.")

    cfg = hydra.utils.instantiate(cfg)

    setup_device(multi_gpu=core_utils.get_param(cfg, "multi_gpu", MultiGPUMode.OFF), num_gpus=core_utils.get_param(cfg, "num_gpus"), device="cuda")

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
        model_name=cfg.architecture,
        num_classes=cfg.arch_params.num_classes,
        arch_params=cfg.arch_params,
        pretrained_weights=cfg.checkpoint_params.pretrained_weights,
        checkpoint_path=cfg.checkpoint_params.checkpoint_path,
        load_backbone=cfg.checkpoint_params.load_backbone,
    ).cuda()  # we assume that QAT is performed with CUDA

    quantize_and_calibrate(
        model,
        train_dataloader,
        method_w=cfg.qat_params.sq_params.method_w,
        method_i=cfg.qat_params.sq_params.method_i,
        per_channel=cfg.qat_params.sq_params.per_channel,
        learn_amax=cfg.qat_params.sq_params.learn_amax,
        skip_modules=cfg.qat_params.sq_params.skip_modules,
        num_calib_batches=cfg.qat_params.calib_params.num_calib_batches or (512 // cfg.dataset_params.train_dataloader_params.batch_size) or 1,
        calibration_method=cfg.qat_params.calib_params.calib_method,
        verbose=cfg.qat_params.verbose,
    )

    if cfg.training_hyperparams.max_epochs != 0:
        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)
        model, _ = trainer.train(model=model, train_loader=train_dataloader, valid_loader=val_dataloader, training_params=cfg.training_hyperparams)
        suffix = "qat"
    else:
        logger.info("cfg.training_hyperparams.max_epochs is 0! Performing PTQ only!")
        suffix = "ptq"

    input_shape = next(iter(val_dataloader))[0].shape
    checkpoints_dir = Path(get_checkpoints_dir_path(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir))
    qat_path = str(checkpoints_dir / f"{cfg.experiment_name}_{'x'.join((str(x) for x in input_shape))}_{suffix}.onnx")
    export_quantized_module_to_onnx(
        model=model.cpu(),
        onnx_filename=qat_path,
        input_shape=input_shape,
        input_size=input_shape,
        train=False,
    )
    logger.info(f"Exporting QAT ONNX to {qat_path}")


if __name__ == "__main__":
    init_trainer()
    main()
