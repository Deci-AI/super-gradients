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
    def __init__(self, batch_size_start: int = 4096, size_step: int = 1024, num_forward_passes: int = 3):
        self.batch_size_start = batch_size_start
        self.size_step = size_step
        self.num_forward_passes = num_forward_passes

    def __call__(self, cfg: DictConfig) -> DictConfig:

        # IMPORT IS HERE DUE TO CIRCULAR IMPORT PROBLEM
        from super_gradients.training.sg_trainer import Trainer

        curr_batch_size = self.batch_size_start
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
        tmp_cfg.training_hyperparams.max_forward_passes_train = self.num_forward_passes
        tmp_cfg.training_hyperparams.run_validation_freq = 2
        tmp_cfg.training_hyperparams.silent_mode = True
        tmp_cfg.training_hyperparams.save_model = False
        tmp_cfg.training_hyperparams.max_epochs = 1
        tmp_cfg.training_hyperparams.average_best_models = False
        tmp_cfg.training_hyperparams.kil_ddp_pgroup_on_end = False
        tmp_cfg.pre_launch_callbacks_list = []

        while True:
            tmp_cfg.dataset_params.train_dataloader_params.batch_size = curr_batch_size

            try:
                Trainer.train_from_config(tmp_cfg)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if curr_batch_size == self.batch_size_start:
                        logger.error("Ran out of memory for the smallest batch, try setting smaller batch_size_start.")
                        raise e
                    else:
                        logger.info(f"Ran out of memory for {curr_batch_size}, setting batch size to {curr_batch_size - self.size_step}.")
                        cfg.dataset_params.train_dataloader_params.batch_size = curr_batch_size - self.size_step
                        for p in model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()

                        # WAIT FOR ALL PROCESSES TO CLEAR THEIR MEMORY BEFORE MOVING ON
                        if is_distributed():
                            barrier()
                        return cfg
                else:
                    raise e

            else:
                logger.info(f"Did not run out of memory for {curr_batch_size}, retrying batch {curr_batch_size + self.size_step}.")
                curr_batch_size += self.size_step
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

                # WAIT FOR ALL PROCESSES TO CLEAR THEIR MEMORY BEFORE MOVING ON
                if is_distributed():
                    barrier()
