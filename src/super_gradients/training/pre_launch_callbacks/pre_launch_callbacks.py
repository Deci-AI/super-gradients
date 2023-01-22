from copy import deepcopy
from typing import Union

from omegaconf import DictConfig
import torch

from super_gradients import is_distributed
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import models
from torch.distributed import barrier

logger = get_logger(__name__)


class PreLaunchCallback:
    """
    PreLaunchCallback

    Base class for callbacks to be triggered, manipulating the config (cfg) prior to launching training,
     when calling Trainer.train_from_config(cfg).

    """

    def __call__(self, cfg: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        raise NotImplementedError


class AutoTrainBatchSizeSelectionCallback(PreLaunchCallback):
    """
    AutoTrainBatchSizeSelectionCallback

    Modifies cfg.dataset_params.train_dataloader_params.batch_size by searching for the maximal batch size that fits
     gpu memory. Works out of the box for DDP.

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
    """

    def __init__(self, min_batch_size: int, size_step: int, num_forward_passes: int = 3, max_batch_size=None, scale_lr: bool = True):
        self.scale_lr = scale_lr
        self.min_batch_size = min_batch_size
        self.size_step = size_step
        self.max_batch_size = max_batch_size
        self.num_forward_passes = num_forward_passes

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

        while True:
            tmp_cfg.dataset_params.train_dataloader_params.batch_size = curr_batch_size

            try:
                Trainer.train_from_config(tmp_cfg)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if curr_batch_size == self.min_batch_size:
                        logger.error("Ran out of memory for the smallest batch, try setting smaller min_batch_size.")
                        raise e
                    else:
                        logger.info(f"Ran out of memory for {curr_batch_size}, setting batch size to {curr_batch_size - self.size_step}.")
                        self._adapt_lr_if_needed(cfg, found_batch_size=curr_batch_size - self.size_step)
                        cfg.dataset_params.train_dataloader_params.batch_size = curr_batch_size - self.size_step
                        self._clear_model_gpu_mem(model)
                        return cfg
                else:
                    raise e

            else:
                if self.max_batch_size is not None and curr_batch_size >= self.max_batch_size:
                    logger.info(
                        f"Did not run out of memory for {curr_batch_size} >= max_batch_size={self.max_batch_size}, " f"setting batch to {self.max_batch_size}."
                    )
                    self._adapt_lr_if_needed(cfg, found_batch_size=self.max_batch_size)
                    cfg.dataset_params.train_dataloader_params.batch_size = self.max_batch_size
                    self._clear_model_gpu_mem(model)
                    return cfg
                logger.info(f"Did not run out of memory for {curr_batch_size}, retrying batch {curr_batch_size + self.size_step}.")
                curr_batch_size += self.size_step
                self._clear_model_gpu_mem(model)

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
