import copy
from copy import deepcopy
from typing import Union

from omegaconf import DictConfig
import torch

from super_gradients.common.environment.cfg_utils import load_recipe
from super_gradients.common.registry.registry import register_pre_launch_callback
from super_gradients import is_distributed
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import models
from torch.distributed import barrier
import cv2
import numpy as np

from super_gradients.training.utils import get_param

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
        tmp_cfg.training_hyperparamsbatch_accumulate = 1
        tmp_cfg.training_hyperparamsmax_train_batches = self.num_forward_passes
        tmp_cfg.training_hyperparamsrun_validation_freq = 2
        tmp_cfg.training_hyperparamssilent_mode = True
        tmp_cfg.training_hyperparamssave_model = False
        tmp_cfg.training_hyperparamsmax_epochs = 1
        tmp_cfg.training_hyperparamsaverage_best_models = False
        tmp_cfg.training_hyperparamskill_ddp_pgroup_on_end = False
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


def modify_params_for_qat(
    training_hyperparams,
    train_dataset_params,
    val_dataset_params,
    train_dataloader_params,
    val_dataloader_params,
    quantization_params=None,
    batch_size_divisor: int = 2,
    max_epochs_divisor: int = 10,
    lr_decay_factor: float = 0.01,
    warmup_epochs_divisor: int = 10,
    cosine_final_lr_ratio: float = 0.01,
    disable_phase_callbacks: bool = True,
    disable_augmentations: bool = False,
):
    """

    This method modifies the recipe for QAT to implement rules of thumb based on the regular non-qat recipe.
    It does so by manipulating the training_hyperparams, train_dataloader_params, val_dataloader_params, train_dataset_params, val_dataset_params.
    Usage:
        trainer = Trainer("test_launch_qat_with_minimal_changes")
        net = ResNet18(num_classes=10, arch_params={})
        train_params = {...}

        train_dataset_params = {
            "transforms": [...
            ]
        }

        train_dataloader_params = {"batch_size": 256}

        val_dataset_params = {"transforms": [ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]}

        val_dataloader_params = {"batch_size": 256}

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params = modify_params_for_qat(
            train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
        )

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.qat(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
            calib_loader=train_loader,
        )

    :param val_dataset_params: Dict, validation dataset_params to be passed to dataloaders.get(...) when instantiating the train dataloader.
    :param train_dataset_params: Dict, train dataset_params to be passed to dataloaders.get(...) when instantiating the validation dataloader.
    :param val_dataloader_params: Dict, validation dataloader_params to be passed to dataloaders.get(...) when instantiating the validation dataloader.
    :param train_dataloader_params: Dict, train dataloader_params to be passed to dataloaders.get(...) when instantiating the train dataloader.
    :param training_hyperparams: Dict, train parameters passed to Trainer.qat(...)
    :param quantization_params: Dict, quantization parameters as passed to Trainer.qat(...). When None, will use the
     default parameters in super_gradients/recipes/quantization_params/default_quantization_params.yaml
    :param int batch_size_divisor: Divisor used to calculate the batch size. Default value is 2.
    :param int max_epochs_divisor: Divisor used to calculate the maximum number of epochs. Default value is 10.
    :param float lr_decay_factor: Factor used to decay the learning rate, weight decay and warmup. Default value is 0.01.
    :param int warmup_epochs_divisor: Divisor used to calculate the number of warm-up epochs. Default value is 10.
    :param float cosine_final_lr_ratio: Ratio used to determine the final learning rate in a cosine annealing schedule. Default value is 0.01.
    :param bool disable_phase_callbacks: Flag to control to disable phase callbacks, which can interfere with QAT. Default value is True.
    :param bool disable_augmentations: Flag to control to disable phase augmentations, which can interfere with QAT. Default value is False.
    :return: modified (copy) training_hyperparams, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
    """
    if quantization_params is None:
        quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params

    quantization_params = deepcopy(quantization_params)
    training_hyperparams = deepcopy(training_hyperparams)
    train_dataloader_params = deepcopy(train_dataloader_params)
    val_dataloader_params = deepcopy(val_dataloader_params)
    train_dataset_params = deepcopy(train_dataset_params)
    val_dataset_params = deepcopy(val_dataset_params)

    if "max_epochs" not in training_hyperparams.keys():
        raise ValueError("max_epochs is a required field in training_hyperparams for QAT modification.")

    if "initial_lr" not in training_hyperparams.keys():
        raise ValueError("initial_lr is a required field in training_hyperparams for QAT modification.")

    if "optimizer_params" not in training_hyperparams.keys():
        raise ValueError("optimizer_params is a required field in training_hyperparams for QAT modification.")

    if "weight_decay" not in training_hyperparams["optimizer_params"].keys():
        raise ValueError("weight_decay is a required field in training_hyperparams['optimizer_params'] for QAT modification.")

    # Q/DQ Layers take a lot of space for activations in training mode
    if get_param(quantization_params, "selective_quantizer_params") and get_param(quantization_params["selective_quantizer_params"], "learn_amax"):
        train_dataloader_params["batch_size"] //= batch_size_divisor
        val_dataloader_params["batch_size"] //= batch_size_divisor

        logger.warning(f"New dataset_params.train_dataloader_params.batch_size: {train_dataloader_params['batch_size']}")
        logger.warning(f"New dataset_params.val_dataloader_params.batch_size: {val_dataloader_params['batch_size']}")
    training_hyperparams["max_epochs"] //= max_epochs_divisor
    logger.warning(f"New number of epochs: {training_hyperparams['max_epochs']}")
    training_hyperparams["initial_lr"] *= lr_decay_factor
    if get_param(training_hyperparams, "warmup_initial_lr") is not None:
        training_hyperparams["warmup_initial_lr"] *= lr_decay_factor
    else:
        training_hyperparams["warmup_initial_lr"] = training_hyperparams["initial_lr"] * 0.01
    training_hyperparams["optimizer_params"]["weight_decay"] *= lr_decay_factor
    logger.warning(f"New learning rate: {training_hyperparams['initial_lr']}")
    logger.warning(f"New weight decay: {training_hyperparams['optimizer_params']['weight_decay']}")
    # as recommended by pytorch-quantization docs
    if get_param(training_hyperparams, "lr_mode") != "cosine":
        training_hyperparams["lr_mode"] = "cosine"
    training_hyperparams["cosine_final_lr_ratio"] = cosine_final_lr_ratio
    logger.warning(
        f"lr_mode will be set to cosine for QAT run instead of {get_param(training_hyperparams, 'lr_mode')} with "
        f"cosine_final_lr_ratio={cosine_final_lr_ratio}"
    )

    training_hyperparams["lr_warmup_epochs"] = (training_hyperparams["max_epochs"] // warmup_epochs_divisor) or 1
    logger.warning(f"New lr_warmup_epochs: {training_hyperparams['lr_warmup_epochs']}")

    # do mess with Q/DQ
    if get_param(training_hyperparams, "ema"):
        logger.warning("EMA will be disabled for QAT run.")
        training_hyperparams["ema"] = False
    if get_param(training_hyperparams, "sync_bn"):
        logger.warning("SyncBatchNorm will be disabled for QAT run.")
        training_hyperparams["sync_bn"] = False
    if disable_phase_callbacks and get_param(training_hyperparams, "phase_callbacks") is not None and len(training_hyperparams["phase_callbacks"]) > 0:
        logger.warning(f"Recipe contains {len(training_hyperparams['phase_callbacks'])} phase callbacks. All of them will be disabled.")
        training_hyperparams["phase_callbacks"] = []
    # no augmentations
    if disable_augmentations and "transforms" in val_dataset_params:
        logger.warning("Augmentations will be disabled for QAT run. Using validation transforms instead.")
        train_dataset_params["transforms"] = val_dataset_params["transforms"]

    return training_hyperparams, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params


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

    USE THIS CALLBACK ONLY WITH Trainer.quantize_from_config
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

        (
            cfg.training_hyperparams,
            cfg.dataset_params.train_dataset_params,
            cfg.dataset_params.val_dataset_params,
            cfg.dataset_params.train_dataloader_params,
            cfg.dataset_params.val_dataloader_params,
        ) = modify_params_for_qat(
            training_hyperparams=cfg.training_hyperparams,
            train_dataset_params=cfg.dataset_params.train_dataset_params,
            train_dataloader_params=cfg.dataset_params.train_dataloader_params,
            val_dataset_params=cfg.dataset_params.val_dataset_params,
            val_dataloader_params=cfg.dataset_params.val_dataloader_params,
            quantization_params=cfg.quantization_params,
            batch_size_divisor=self.batch_size_divisor,
            disable_phase_callbacks=self.disable_phase_callbacks,
            cosine_final_lr_ratio=self.cosine_final_lr_ratio,
            warmup_epochs_divisor=self.warmup_epochs_divisor,
            lr_decay_factor=self.lr_decay_factor,
            max_epochs_divisor=self.max_epochs_divisor,
            disable_augmentations=self.disable_augmentations,
        )

        if cfg.multi_gpu != "OFF" or cfg.num_gpus != 1:
            logger.warning(f"Recipe requests multi_gpu={cfg.multi_gpu} and num_gpus={cfg.num_gpus}. Changing to multi_gpu=OFF and num_gpus=1")
            cfg.multi_gpu = "OFF"
            cfg.num_gpus = 1

        return cfg
