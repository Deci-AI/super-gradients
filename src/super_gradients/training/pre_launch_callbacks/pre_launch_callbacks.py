import copy
from copy import deepcopy
from typing import Union

from omegaconf import DictConfig
import torch

from super_gradients.common.registry.registry import register_pre_launch_callback
from super_gradients import is_distributed
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import models
from torch.distributed import barrier
import cv2
import numpy as np

logger = get_logger(__name__)


class PreLaunchCallback:
    """
    PreLaunchCallback

    Base class for callbacks to be triggered, manipulating the config (cfg) prior to launching training,
     when calling Trainer.train_from_config(cfg).

    """

    def __call__(self, cfg: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        raise NotImplementedError


@register_pre_launch_callback()
class AutoTrainBatchSizeSelectionCallback(PreLaunchCallback):
    """
    AutoTrainBatchSizeSelectionCallback

    Modifies cfg.dataset_params.train_dataloader_params.batch_size by searching for the maximal batch size that fits
     gpu memory/ the one resulting in fastest time for the selected number of train datalaoder iterations. Works out of the box for DDP.

    The search is done by running a few forward passes for increasing batch sizes, until CUDA OUT OF MEMORY is raised:

        For batch_size in range(min_batch_size:max_batch_size:size_step):
            if batch_size raises CUDA OUT OF MEMORY ERROR:
                return batch_size-size_step
        return batch_size

    Example usage: Inside the main recipe .YAML file (for example super_gradients/recipes/cifar10_resnet.yaml),
     add the following:

    pre_launch_callbacks_list:
        - AutoTrainBatchSizeSelectionCallback:
            min_batch_size: 128
            size_step: 64
            num_forward_passes: 10

    Then, when running super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=...
    this pre_launch_callback will modify cfg.dataset_params.train_dataloader_params.batch_size then pass cfg to
     Trainer.train_from_config(cfg) and training will continue with the selected batch size.


    :param min_batch_size: int, the first batch size to try running forward passes. Should fit memory.

    :param size_step: int, the difference between 2 consecutive batch_size trials.

    :param num_forward_passes: int, number of forward passes (i.e train_loader data iterations inside an epoch).
     Note that the more forward passes being done, the less the selected batch size is prawn to fail. This is because
      other then gradients, model computations, data and other fixed gpu memory that is being used- some more gpu memory
       might be used by the metric objects and PhaseCallbacks.

    :param max_batch_size: int, optional, upper limit of the batch sizes to try. When None, the search will continue until
     the maximal batch size that does not raise CUDA OUT OF MEMORY is found (deafult=None).

    :param scale_lr: bool, whether to linearly scale cfg.training_hyperparams.initial_lr, i.e multiply by
     FOUND_BATCH_SIZE/cfg.dataset_params.train_datalaoder_params.batch_size (default=True)
    :param mode: str, one of ["fastest","largest"], whether to select the largest batch size that fits memory or the one
     that the resulted in overall fastest execution.
    """

    def __init__(self, min_batch_size: int, size_step: int, num_forward_passes: int = 3, max_batch_size=None, scale_lr: bool = True, mode: str = "fastest"):
        if mode not in ["fastest", "largest"]:
            raise TypeError(f"Expected mode to be one of: ['fastest','largest'], got {mode}")
        self.scale_lr = scale_lr
        self.min_batch_size = min_batch_size
        self.size_step = size_step
        self.max_batch_size = max_batch_size
        self.num_forward_passes = num_forward_passes
        self.mode = mode

    def __call__(self, cfg: DictConfig) -> DictConfig:

        # IMPORT IS HERE DUE TO CIRCULAR IMPORT PROBLEM
        from super_gradients.training.sg_trainer import Trainer

        curr_batch_size = self.min_batch_size
        # BUILD NETWORK
        model = models.get(
            model_name=cfg.architecture,
            num_classes=cfg.arch_params.num_classes,
            arch_params=cfg.arch_params,
            strict_load=cfg.checkpoint_params.strict_load,
            pretrained_weights=cfg.checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.checkpoint_params.checkpoint_path,
            load_backbone=cfg.checkpoint_params.load_backbone,
        )
        tmp_cfg = deepcopy(cfg)
        tmp_cfg.training_hyperparams.batch_accumulate = 1
        tmp_cfg.training_hyperparams.max_train_batches = self.num_forward_passes
        tmp_cfg.training_hyperparams.run_validation_freq = 2
        tmp_cfg.training_hyperparams.silent_mode = True
        tmp_cfg.training_hyperparams.save_model = False
        tmp_cfg.training_hyperparams.max_epochs = 1
        tmp_cfg.training_hyperparams.average_best_models = False
        tmp_cfg.training_hyperparams.kill_ddp_pgroup_on_end = False
        tmp_cfg.pre_launch_callbacks_list = []

        fastest_batch_time = np.inf
        fastest_batch_size = curr_batch_size

        bs_found = False

        while not bs_found:
            tmp_cfg.dataset_params.train_dataloader_params.batch_size = curr_batch_size

            try:
                passes_start = cv2.getTickCount()
                Trainer.train_from_config(tmp_cfg)
                curr_batch_time = (cv2.getTickCount() - passes_start) / cv2.getTickFrequency()
                logger.info(f"Batch size = {curr_batch_size} time for {self.num_forward_passes} forward passes: {curr_batch_time} seconds.")
                if curr_batch_time < fastest_batch_time:
                    fastest_batch_size = curr_batch_size
                    fastest_batch_time = curr_batch_time

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if curr_batch_size == self.min_batch_size:
                        logger.error("Ran out of memory for the smallest batch, try setting smaller min_batch_size.")
                        raise e
                    else:
                        selected_batch_size = curr_batch_size - self.size_step if self.mode == "largest" else fastest_batch_size
                        msg = f"Ran out of memory for {curr_batch_size}, setting batch size to {selected_batch_size}."
                        bs_found = True
                else:
                    raise e

            else:
                if self.max_batch_size is not None and curr_batch_size >= self.max_batch_size:
                    selected_batch_size = self.max_batch_size if self.mode == "largest" else fastest_batch_size
                    msg = (
                        f"Did not run out of memory for {curr_batch_size} >= max_batch_size={self.max_batch_size}, " f"setting batch to {selected_batch_size}."
                    )
                    bs_found = True
                else:
                    logger.info(f"Did not run out of memory for {curr_batch_size}, retrying batch {curr_batch_size + self.size_step}.")
                    curr_batch_size += self.size_step
                    self._clear_model_gpu_mem(model)

        return self._inject_selected_batch_size_to_config(cfg, model, msg, selected_batch_size)

    def _inject_selected_batch_size_to_config(self, cfg, model, msg, selected_batch_size):
        logger.info(msg)
        self._adapt_lr_if_needed(cfg, found_batch_size=selected_batch_size)
        cfg.dataset_params.train_dataloader_params.batch_size = selected_batch_size
        self._clear_model_gpu_mem(model)
        return cfg

    def _adapt_lr_if_needed(self, cfg: DictConfig, found_batch_size: int) -> DictConfig:
        if self.scale_lr:
            scale_factor = found_batch_size / cfg.dataset_params.train_dataloader_params.batch_size
            cfg.training_hyperparams.initial_lr = cfg.training_hyperparams.initial_lr * scale_factor
        return cfg

    @classmethod
    def _clear_model_gpu_mem(cls, model):
        for p in model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()
        # WAIT FOR ALL PROCESSES TO CLEAR THEIR MEMORY BEFORE MOVING ON
        if is_distributed():
            barrier()


@register_pre_launch_callback()
class QATRecipeModificationCallback(PreLaunchCallback):
    """
     QATRecipeModificationCallback(PreLaunchCallback)

    This callback modifies the recipe for QAT to implement rules of thumb based on the regular non-qat recipe.

    :param int batch_size_divisor: Divisor used to calculate the batch size. Default value is 2.
    :param int max_epochs_divisor: Divisor used to calculate the maximum number of epochs. Default value is 10.
    :param float lr_decay_factor: Factor used to decay the learning rate, weight decay and warmup. Default value is 0.01.
    :param int warmup_epochs_divisor: Divisor used to calculate the number of warm-up epochs. Default value is 10.
    :param float cosine_final_lr_ratio: Ratio used to determine the final learning rate in a cosine annealing schedule. Default value is 0.01.
    :param bool disable_phase_callbacks: Flag to control to disable phase callbacks, which can interfere with QAT. Default value is True.
    :param bool disable_augmentations: Flag to control to disable phase augmentations, which can interfere with QAT. Default value is False.

    Example usage:

    Inside the main recipe .YAML file (for example super_gradients/recipes/cifar10_resnet.yaml), add the following:

    pre_launch_callbacks_list:
        - QATRecipeModificationCallback:
            batch_size_divisor: 2
            max_epochs_divisor: 10
            lr_decay_factor: 0.01
            warmup_epochs_divisor: 10
            cosine_final_lr_ratio: 0.01
            disable_phase_callbacks: True
            disable_augmentations: False

    USE THIS CALLBACK ONLY WITH QATTrainer!
    """

    def __init__(
        self,
        batch_size_divisor: int = 2,
        max_epochs_divisor: int = 10,
        lr_decay_factor: float = 0.01,
        warmup_epochs_divisor: int = 10,
        cosine_final_lr_ratio: float = 0.01,
        disable_phase_callbacks: bool = True,
        disable_augmentations: bool = False,
    ):
        self.disable_augmentations = disable_augmentations
        self.disable_phase_callbacks = disable_phase_callbacks
        self.cosine_final_lr_ratio = cosine_final_lr_ratio
        self.warmup_epochs_divisor = warmup_epochs_divisor
        self.lr_decay_factor = lr_decay_factor
        self.max_epochs_divisor = max_epochs_divisor
        self.batch_size_divisor = batch_size_divisor

    def __call__(self, cfg: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        logger.info("Modifying recipe to suit QAT rules of thumb. Remove QATRecipeModificationCallback to disable.")

        cfg = copy.deepcopy(cfg)

        # Q/DQ Layers take a lot of space for activations in training mode
        if cfg.quantization_params.selective_quantizer_params.learn_amax:
            cfg.dataset_params.train_dataloader_params.batch_size //= self.batch_size_divisor
            cfg.dataset_params.val_dataloader_params.batch_size //= self.batch_size_divisor

            logger.warning(f"New dataset_params.train_dataloader_params.batch_size: {cfg.dataset_params.train_dataloader_params.batch_size}")
            logger.warning(f"New dataset_params.val_dataloader_params.batch_size: {cfg.dataset_params.val_dataloader_params.batch_size}")

        cfg.training_hyperparams.max_epochs //= self.max_epochs_divisor
        logger.warning(f"New number of epochs: {cfg.training_hyperparams.max_epochs}")

        cfg.training_hyperparams.initial_lr *= self.lr_decay_factor
        if cfg.training_hyperparams.warmup_initial_lr is not None:
            cfg.training_hyperparams.warmup_initial_lr *= self.lr_decay_factor
        else:
            cfg.training_hyperparams.warmup_initial_lr = cfg.training_hyperparams.initial_lr * 0.01

        cfg.training_hyperparams.optimizer_params.weight_decay *= self.lr_decay_factor

        logger.warning(f"New learning rate: {cfg.training_hyperparams.initial_lr}")
        logger.warning(f"New weight decay: {cfg.training_hyperparams.optimizer_params.weight_decay}")

        # as recommended by pytorch-quantization docs
        cfg.training_hyperparams.lr_mode = "cosine"
        cfg.training_hyperparams.lr_warmup_epochs = (cfg.training_hyperparams.max_epochs // self.warmup_epochs_divisor) or 1
        cfg.training_hyperparams.cosine_final_lr_ratio = self.cosine_final_lr_ratio

        # do mess with Q/DQ
        if cfg.training_hyperparams.ema:
            logger.warning("EMA will be disabled for QAT run.")
            cfg.training_hyperparams.ema = False

        if cfg.training_hyperparams.sync_bn:
            logger.warning("SyncBatchNorm will be disabled for QAT run.")
            cfg.training_hyperparams.sync_bn = False

        if self.disable_phase_callbacks and len(cfg.training_hyperparams.phase_callbacks) > 0:
            logger.warning(f"Recipe contains {len(cfg.training_hyperparams.phase_callbacks)} phase callbacks. All of them will be disabled.")
            cfg.training_hyperparams.phase_callbacks = []

        if cfg.multi_gpu != "OFF" or cfg.num_gpus != 1:
            logger.warning(f"Recipe requests multi_gpu={cfg.multi_gpu} and num_gpus={cfg.num_gpus}. Changing to multi_gpu=OFF and num_gpus=1")
            cfg.multi_gpu = "OFF"
            cfg.num_gpus = 1

        # no augmentations
        if self.disable_augmentations and "transforms" in cfg.dataset_params.val_dataset_params:
            logger.warning("Augmentations will be disabled for QAT run.")
            cfg.dataset_params.train_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms

        return cfg
