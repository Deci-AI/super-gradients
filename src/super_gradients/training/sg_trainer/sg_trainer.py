import copy
import inspect
import os
from copy import deepcopy
from pathlib import Path
from typing import Union, Tuple, Mapping, Dict, Any, List

import hydra
import numpy as np
import torch
import torch.cuda
import torch.nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from piptools.scripts.sync import _get_installed_distributions
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MetricCollection, Metric
from tqdm import tqdm

from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path, get_ckpt_local_path
from super_gradients.module_interfaces import HasPreprocessingParams, HasPredict
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches

from super_gradients.training.utils.sg_trainer_utils import get_callable_param_names
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.data_types.enum import MultiGPUMode, StrictLoad, EvaluationType
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.callbacks_factory import CallbacksFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.losses_factory import LossesFactory
from super_gradients.common.factories.metrics_factory import MetricsFactory

from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.datasets.samplers import RepeatAugSampler
from super_gradients.training.exceptions.sg_trainer_exceptions import UnsupportedOptimizerFormat
from super_gradients.training.metrics.metric_utils import (
    get_metrics_titles,
    get_metrics_results_tuple,
    get_logging_values,
    get_metrics_dict,
    get_train_loop_description_dict,
)
from super_gradients.training.models import SgModule, get_model_name
from super_gradients.common.registry.registry import ARCHITECTURES, SG_LOGGERS
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import sg_trainer_utils, get_param
from super_gradients.training.utils.distributed_training_utils import (
    MultiGPUModeAutocastWrapper,
    reduce_results_tuple_for_ddp,
    compute_precise_bn_stats,
    setup_device,
    get_gpu_mem_utilization,
    get_world_size,
    get_local_rank,
    require_ddp_setup,
    get_device_ids,
    is_ddp_subprocess,
    wait_for_the_master,
    DDPNotSetupException,
)
from super_gradients.training.utils.ema import ModelEMA
from super_gradients.training.utils.optimizer_utils import build_optimizer
from super_gradients.training.utils.sg_trainer_utils import MonitoredValue, log_main_training_params
from super_gradients.training.utils.utils import fuzzy_idx_in_list
from super_gradients.training.utils.weight_averaging_utils import ModelWeightAveraging
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils import random_seed
from super_gradients.training.utils.checkpoint_utils import (
    read_ckpt_state_dict,
    load_checkpoint_to_model,
    load_pretrained_weights,
)
from super_gradients.training.datasets.datasets_utils import DatasetStatisticsTensorboardLogger
from super_gradients.training.utils.callbacks import (
    CallbackHandler,
    Phase,
    PhaseContext,
    MetricsUpdateCallback,
    ContextSgMethods,
    LRCallbackBase,
)
from super_gradients.common.registry.registry import LR_SCHEDULERS_CLS_DICT, LR_WARMUP_CLS_DICT
from super_gradients.common.environment.device_utils import device_config
from super_gradients.training.utils import HpmStruct
from super_gradients.common.environment.cfg_utils import load_experiment_cfg, add_params_to_cfg, load_recipe
from super_gradients.common.factories.pre_launch_callbacks_factory import PreLaunchCallbacksFactory
from super_gradients.training.params import TrainingParams

logger = get_logger(__name__)


try:
    from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
    from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
    from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

    _imported_pytorch_quantization_failure = None

except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug("Failed to import pytorch_quantization:")
    logger.debug(import_err)
    _imported_pytorch_quantization_failure = import_err


class Trainer:
    """
    SuperGradient Model - Base Class for Sg Models

    Methods
    -------
    train(max_epochs : int, initial_epoch : int, save_model : bool)
        the main function used for the training, h.p. updating, logging etc.

    predict(idx : int)
        returns the predictions and label of the current inputs

    test(epoch : int, idx : int, save : bool):
        returns the test loss, accuracy and runtime
    """

    def __init__(self, experiment_name: str, device: str = None, multi_gpu: Union[MultiGPUMode, str] = None, ckpt_root_dir: str = None):
        """

        :param experiment_name:                      Used for logging and loading purposes
        :param device:                          If equal to 'cpu' runs on the CPU otherwise on GPU
        :param multi_gpu:                       If True, runs on all available devices
                                                otherwise saves the Checkpoints Locally
                                                checkpoint from cloud service, otherwise overwrites the local checkpoints file
        :param ckpt_root_dir:                   Local root directory path where all experiment logging directories will
                                                reside. When none is give, it is assumed that
                                                pkg_resources.resource_filename('checkpoints', "") exists and will be used.

        """

        # This should later me removed
        if device is not None or multi_gpu is not None:
            raise KeyError(
                "Trainer does not accept anymore 'device' and 'multi_gpu' as argument. "
                "Both should instead be passed to "
                "super_gradients.setup_device(device=..., multi_gpu=..., num_gpus=...)"
            )

        if require_ddp_setup():
            raise DDPNotSetupException()

        # SET THE EMPTY PROPERTIES
        self.net, self.architecture, self.arch_params, self.dataset_interface = None, None, None, None
        self.train_loader, self.valid_loader = None, None
        self.ema = None
        self.ema_model = None
        self.sg_logger = None
        self.update_param_groups = None
        self.criterion = None
        self.training_params = None
        self.scaler = None
        self.phase_callbacks = None
        self.checkpoint_params = None
        self.pre_prediction_callback = None

        # SET THE DEFAULT PROPERTIES
        self.half_precision = False
        self.load_checkpoint = False
        self.load_backbone = False
        self.load_weights_only = False
        self.ddp_silent_mode = is_ddp_subprocess()

        self.model_weight_averaging = None
        self.average_model_checkpoint_filename = "average_model.pth"
        self.start_epoch = 0
        self.best_metric = np.inf
        self.external_checkpoint_path = None
        self.strict_load = StrictLoad.ON
        self.load_ema_as_net = False
        self.ckpt_best_name = "ckpt_best.pth"
        self._first_backward = True

        # METRICS
        self.loss_logging_items_names = None
        self.train_metrics = None
        self.valid_metrics = None
        self.greater_metric_to_watch_is_better = None
        self.metric_to_watch = None
        self.greater_train_metrics_is_better: Dict[str, bool] = {}  # For each metric, indicates if greater is better
        self.greater_valid_metrics_is_better: Dict[str, bool] = {}

        # SETTING THE PROPERTIES FROM THE CONSTRUCTOR
        self.experiment_name = experiment_name
        self.ckpt_name = None

        self.checkpoints_dir_path = get_checkpoints_dir_path(experiment_name, ckpt_root_dir)
        self.phase_callback_handler: CallbackHandler = None

        # SET THE DEFAULTS
        # TODO: SET DEFAULT TRAINING PARAMS FOR EACH TASK

        default_results_titles = ["Train Loss", "Train Acc", "Train Top5", "Valid Loss", "Valid Acc", "Valid Top5"]

        self.results_titles = default_results_titles

        default_train_metrics, default_valid_metrics = MetricCollection([Accuracy(), Top5()]), MetricCollection([Accuracy(), Top5()])

        self.train_metrics, self.valid_metrics = default_train_metrics, default_valid_metrics

        self.train_monitored_values = {}
        self.valid_monitored_values = {}
        self.max_train_batches = None
        self.max_valid_batches = None

        self._epoch_start_logging_values = {}

    @property
    def device(self) -> str:
        return device_config.device

    @classmethod
    def train_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:
        """
        Trains according to cfg recipe configuration.

        :param cfg: The parsed DictConfig from yaml recipe files or a dictionary
        :return: the model and the output of trainer.train(...) (i.e results tuple)
        """

        setup_device(
            device=core_utils.get_param(cfg, "device"),
            multi_gpu=core_utils.get_param(cfg, "multi_gpu"),
            num_gpus=core_utils.get_param(cfg, "num_gpus"),
        )

        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        # TRIGGER CFG MODIFYING CALLBACKS
        cfg = cls._trigger_cfg_modifying_callbacks(cfg)

        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)

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

        # INSTANTIATE DATA LOADERS

        train_dataloader = dataloaders.get(
            name=get_param(cfg, "train_dataloader"),
            dataset_params=cfg.dataset_params.train_dataset_params,
            dataloader_params=cfg.dataset_params.train_dataloader_params,
        )

        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=cfg.dataset_params.val_dataset_params,
            dataloader_params=cfg.dataset_params.val_dataloader_params,
        )

        recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}
        # TRAIN
        res = trainer.train(
            model=model,
            train_loader=train_dataloader,
            valid_loader=val_dataloader,
            training_params=cfg.training_hyperparams,
            additional_configs_to_log=recipe_logged_cfg,
        )

        return model, res

    @classmethod
    def _trigger_cfg_modifying_callbacks(cls, cfg):
        pre_launch_cbs = get_param(cfg, "pre_launch_callbacks_list", list())
        pre_launch_cbs = ListFactory(PreLaunchCallbacksFactory()).get(pre_launch_cbs)
        for plcb in pre_launch_cbs:
            cfg = plcb(cfg)
        return cfg

    @classmethod
    def resume_experiment(cls, experiment_name: str, ckpt_root_dir: str = None) -> Tuple[nn.Module, Tuple]:
        """
        Resume a training that was run using our recipes.

        :param experiment_name:     Name of the experiment to resume
        :param ckpt_root_dir:       Directory including the checkpoints
        """
        logger.info("Resume training using the checkpoint recipe, ignoring the current recipe")
        cfg = load_experiment_cfg(experiment_name, ckpt_root_dir)
        add_params_to_cfg(cfg, params=["training_hyperparams.resume=True"])
        return cls.train_from_config(cfg)

    @classmethod
    def evaluate_from_recipe(cls, cfg: DictConfig) -> Tuple[nn.Module, Tuple]:
        """
        Evaluate according to a cfg recipe configuration.

        Note:   This script does NOT run training, only validation.
                Please make sure that the config refers to a PRETRAINED MODEL either from one of your checkpoint or from pretrained weights from model zoo.
        :param cfg: The parsed DictConfig from yaml recipe files or a dictionary
        """

        setup_device(
            device=core_utils.get_param(cfg, "device"),
            multi_gpu=core_utils.get_param(cfg, "multi_gpu"),
            num_gpus=core_utils.get_param(cfg, "num_gpus"),
        )

        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)

        # INSTANTIATE DATA LOADERS
        val_dataloader = dataloaders.get(
            name=cfg.val_dataloader, dataset_params=cfg.dataset_params.val_dataset_params, dataloader_params=cfg.dataset_params.val_dataloader_params
        )

        if cfg.checkpoint_params.checkpoint_path is None:
            logger.info(
                "checkpoint_params.checkpoint_path was not provided, " "so the recipe will be evaluated using checkpoints_dir/training_hyperparams.ckpt_name"
            )
            checkpoints_dir = get_checkpoints_dir_path(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)
            checkpoint_path = os.path.join(checkpoints_dir, cfg.training_hyperparams.ckpt_name)
            if os.path.exists(checkpoint_path):
                cfg.checkpoint_params.checkpoint_path = checkpoint_path

        logger.info(f"Evaluating checkpoint: {cfg.checkpoint_params.checkpoint_path}")

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

        # TEST
        valid_metrics_dict = trainer.test(model=model, test_loader=val_dataloader, test_metrics_list=cfg.training_hyperparams.valid_metrics_list)

        results = ["Validate Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in valid_metrics_dict.items()]
        logger.info("\n".join(results))

        return model, valid_metrics_dict

    @classmethod
    def evaluate_checkpoint(cls, experiment_name: str, ckpt_name: str = "ckpt_latest.pth", ckpt_root_dir: str = None) -> None:
        """
        Evaluate a checkpoint resulting from one of your previous experiment, using the same parameters (dataset, valid_metrics,...)
        as used during the training of the experiment

        Note:
            The parameters will be unchanged even if the recipe used for that experiment was changed since then.
            This is to ensure that validation of the experiment will remain exactly the same as during training.

        Example, evaluate the checkpoint "average_model.pth" from experiment "my_experiment_name":
            >> evaluate_checkpoint(experiment_name="my_experiment_name", ckpt_name="average_model.pth")

        :param experiment_name:     Name of the experiment to validate
        :param ckpt_name:           Name of the checkpoint to test ("ckpt_latest.pth", "average_model.pth" or "ckpt_best.pth" for instance)
        :param ckpt_root_dir:       Directory including the checkpoints
        """
        logger.info("Evaluate checkpoint")
        cfg = load_experiment_cfg(experiment_name, ckpt_root_dir)
        add_params_to_cfg(cfg, params=["training_hyperparams.resume=True", f"ckpt_name={ckpt_name}"])
        cls.evaluate_from_recipe(cfg)

    def _set_dataset_params(self):
        self.dataset_params = {
            "train_dataset_params": self.train_loader.dataset.dataset_params if hasattr(self.train_loader.dataset, "dataset_params") else None,
            "train_dataloader_params": self.train_loader.dataloader_params if hasattr(self.train_loader, "dataloader_params") else None,
            "valid_dataset_params": self.valid_loader.dataset.dataset_params if hasattr(self.valid_loader.dataset, "dataset_params") else None,
            "valid_dataloader_params": self.valid_loader.dataloader_params if hasattr(self.valid_loader, "dataloader_params") else None,
        }
        self.dataset_params = HpmStruct(**self.dataset_params)

    def _net_to_device(self):
        """
        Manipulates self.net according to device.multi_gpu
        """
        self.net.to(device_config.device)

        # FOR MULTI-GPU TRAINING (not distributed)
        sync_bn = core_utils.get_param(self.training_params, "sync_bn", default_val=False)
        if device_config.multi_gpu == MultiGPUMode.DATA_PARALLEL:
            self.net = torch.nn.DataParallel(self.net, device_ids=get_device_ids())
        elif device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            if sync_bn:
                if not self.ddp_silent_mode:
                    logger.info("DDP - Using Sync Batch Norm... Training time will be affected accordingly")
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net).to(device_config.device)

            local_rank = int(device_config.device.split(":")[1])
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        else:
            self.net = core_utils.WrappedModel(self.net)

    def _train_epoch(self, epoch: int, silent_mode: bool = False) -> tuple:
        """
        train_epoch - A single epoch training procedure
            :param optimizer:   The optimizer for the network
            :param epoch:       The current epoch
            :param silent_mode: No verbosity
        """
        # SET THE MODEL IN training STATE
        self.net.train()

        # THE DISABLE FLAG CONTROLS WHETHER THE PROGRESS BAR IS SILENT OR PRINTS THE LOGS
        with tqdm(self.train_loader, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True, disable=silent_mode) as progress_bar_train_loader:
            progress_bar_train_loader.set_description(f"Train epoch {epoch}")

            # RESET/INIT THE METRIC LOGGERS
            self._reset_metrics()

            self.train_metrics.to(device_config.device)
            loss_avg_meter = core_utils.utils.AverageMeter()

            context = PhaseContext(
                epoch=epoch,
                optimizer=self.optimizer,
                metrics_compute_fn=self.train_metrics,
                loss_avg_meter=loss_avg_meter,
                criterion=self.criterion,
                device=device_config.device,
                lr_warmup_epochs=self.training_params.lr_warmup_epochs,
                sg_logger=self.sg_logger,
                train_loader=self.train_loader,
                valid_loader=self.valid_loader,
                context_methods=self._get_context_methods(Phase.TRAIN_BATCH_END),
                ddp_silent_mode=self.ddp_silent_mode,
            )

            for batch_idx, batch_items in enumerate(progress_bar_train_loader):
                batch_items = core_utils.tensor_container_to_device(batch_items, device_config.device, non_blocking=True)
                inputs, targets, additional_batch_items = sg_trainer_utils.unpack_batch_items(batch_items)

                if self.pre_prediction_callback is not None:
                    inputs, targets = self.pre_prediction_callback(inputs, targets, batch_idx)

                context.update_context(batch_idx=batch_idx, inputs=inputs, target=targets, **additional_batch_items)
                self.phase_callback_handler.on_train_batch_start(context)

                # AUTOCAST IS ENABLED ONLY IF self.training_params.mixed_precision - IF enabled=False AUTOCAST HAS NO EFFECT
                with autocast(enabled=self.training_params.mixed_precision):
                    # FORWARD PASS TO GET NETWORK'S PREDICTIONS
                    outputs = self.net(inputs)

                    # COMPUTE THE LOSS FOR BACK PROP + EXTRA METRICS COMPUTED DURING THE LOSS FORWARD PASS
                    loss, loss_log_items = self._get_losses(outputs, targets)

                context.update_context(preds=outputs, loss_log_items=loss_log_items)
                self.phase_callback_handler.on_train_batch_loss_end(context)

                if not self.ddp_silent_mode and batch_idx == 0:
                    self._epoch_start_logging_values = self._get_epoch_start_logging_values()

                self._backward_step(loss, epoch, batch_idx, context)

                # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
                logging_values = loss_avg_meter.average + get_metrics_results_tuple(self.train_metrics)
                gpu_memory_utilization = get_gpu_mem_utilization() / 1e9 if torch.cuda.is_available() else 0

                # RENDER METRICS PROGRESS
                pbar_message_dict = get_train_loop_description_dict(
                    logging_values, self.train_metrics, self.loss_logging_items_names, gpu_mem=gpu_memory_utilization
                )

                progress_bar_train_loader.set_postfix(**pbar_message_dict)
                self.phase_callback_handler.on_train_batch_end(context)

                if self.max_train_batches is not None and self.max_train_batches - 1 <= batch_idx:
                    break

            self.train_monitored_values = sg_trainer_utils.update_monitored_values_dict(
                monitored_values_dict=self.train_monitored_values, new_values_dict=pbar_message_dict
            )

        return logging_values

    def _get_losses(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        # GET THE OUTPUT OF THE LOSS FUNCTION
        loss = self.criterion(outputs, targets)
        if isinstance(loss, tuple):
            loss, loss_logging_items = loss
            # IF ITS NOT A TUPLE THE LOGGING ITEMS CONTAIN ONLY THE LOSS FOR BACKPROP (USER DEFINED LOSS RETURNS SCALAR)
        else:
            loss_logging_items = loss.unsqueeze(0).detach()

        # ON FIRST BACKWARD, DERRIVE THE LOGGING TITLES.
        if self.loss_logging_items_names is None or self._first_backward:
            self._init_loss_logging_names(loss_logging_items)
            if self.metric_to_watch:
                self._init_monitored_items()
            self._first_backward = False

        if len(loss_logging_items) != len(self.loss_logging_items_names):
            raise ValueError(
                "Loss output length must match loss_logging_items_names. Got "
                + str(len(loss_logging_items))
                + ", and "
                + str(len(self.loss_logging_items_names))
            )
        # RETURN AND THE LOSS LOGGING ITEMS COMPUTED DURING LOSS FORWARD PASS
        return loss, loss_logging_items

    def _init_monitored_items(self):
        self.metric_idx_in_results_tuple = fuzzy_idx_in_list(self.metric_to_watch, self.loss_logging_items_names + get_metrics_titles(self.valid_metrics))
        # Instantiate the values to monitor (loss/metric)
        for loss_name in self.loss_logging_items_names:
            self.train_monitored_values[loss_name] = MonitoredValue(name=loss_name, greater_is_better=False)
            self.valid_monitored_values[loss_name] = MonitoredValue(name=loss_name, greater_is_better=False)

        for metric_name in get_metrics_titles(self.train_metrics):
            self.train_monitored_values[metric_name] = MonitoredValue(name=metric_name, greater_is_better=self.greater_train_metrics_is_better.get(metric_name))

        for metric_name in get_metrics_titles(self.valid_metrics):
            self.valid_monitored_values[metric_name] = MonitoredValue(name=metric_name, greater_is_better=self.greater_valid_metrics_is_better.get(metric_name))

        self.results_titles = ["Train_" + t for t in self.loss_logging_items_names + get_metrics_titles(self.train_metrics)] + [
            "Valid_" + t for t in self.loss_logging_items_names + get_metrics_titles(self.valid_metrics)
        ]

        if self.training_params.average_best_models:
            self.model_weight_averaging = ModelWeightAveraging(
                ckpt_dir=self.checkpoints_dir_path,
                greater_is_better=self.greater_metric_to_watch_is_better,
                metric_to_watch=self.metric_to_watch,
                metric_idx=self.metric_idx_in_results_tuple,
                load_checkpoint=self.load_checkpoint,
            )

    def _backward_step(self, loss: torch.Tensor, epoch: int, batch_idx: int, context: PhaseContext, *args, **kwargs) -> None:
        """
        Run backprop on the loss and perform a step
        :param loss: The value computed by the loss function
        :param optimizer: An object that can perform a gradient step and zeroize model gradient
        :param epoch: number of epoch the training is on
        :param batch_idx: Zero-based number of iteration inside the current epoch
        :param context: current phase context
        :return:
        """
        # SCALER IS ENABLED ONLY IF self.training_params.mixed_precision=True
        self.scaler.scale(loss).backward()
        self.phase_callback_handler.on_train_batch_backward_end(context)

        # ACCUMULATE GRADIENT FOR X BATCHES BEFORE OPTIMIZING
        local_step = batch_idx + 1
        global_step = local_step + len(self.train_loader) * epoch
        total_steps = len(self.train_loader) * self.max_epochs

        if global_step % self.batch_accumulate == 0:
            self.phase_callback_handler.on_train_batch_gradient_step_start(context)

            # APPLY GRADIENT CLIPPING IF REQUIRED
            if self.training_params.clip_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.training_params.clip_grad_norm)

            # SCALER IS ENABLED ONLY IF self.training_params.mixed_precision=True
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()
            if self.ema:
                self.ema_model.update(self.net, step=global_step, total_steps=total_steps)

            # RUN PHASE CALLBACKS
            self.phase_callback_handler.on_train_batch_gradient_step_end(context)

    def _save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = None,
        validation_results_tuple: tuple = None,
        context: PhaseContext = None,
    ) -> None:
        """
        Save the current state dict as latest (always), best (if metric was improved), epoch# (if determined in training
        params)
        """
        # WHEN THE validation_results_tuple IS NONE WE SIMPLY SAVE THE state_dict AS LATEST AND Return
        if validation_results_tuple is None:
            self.sg_logger.add_checkpoint(tag="ckpt_latest_weights_only.pth", state_dict={"net": self.net.state_dict()}, global_step=epoch)
            return

        # COMPUTE THE CURRENT metric
        # IF idx IS A LIST - SUM ALL THE VALUES STORED IN THE LIST'S INDICES
        metric = (
            validation_results_tuple[self.metric_idx_in_results_tuple]
            if isinstance(self.metric_idx_in_results_tuple, int)
            else sum([validation_results_tuple[idx] for idx in self.metric_idx_in_results_tuple])
        )

        # BUILD THE state_dict
        state = {"net": self.net.state_dict(), "acc": metric, "epoch": epoch}
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        if self.ema:
            state["ema_net"] = self.ema_model.ema.state_dict()

        if isinstance(self.net.module, HasPredict) and isinstance(self.valid_loader.dataset, HasPreprocessingParams):
            state["processing_params"] = self.valid_loader.dataset.get_dataset_preprocessing_params()

        # SAVES CURRENT MODEL AS ckpt_latest
        self.sg_logger.add_checkpoint(tag="ckpt_latest.pth", state_dict=state, global_step=epoch)

        # SAVE MODEL AT SPECIFIC EPOCHS DETERMINED BY save_ckpt_epoch_list
        if epoch in self.training_params.save_ckpt_epoch_list:
            self.sg_logger.add_checkpoint(tag=f"ckpt_epoch_{epoch}.pth", state_dict=state, global_step=epoch)

        # OVERRIDE THE BEST CHECKPOINT AND best_metric IF metric GOT BETTER THAN THE PREVIOUS BEST
        if (metric > self.best_metric and self.greater_metric_to_watch_is_better) or (metric < self.best_metric and not self.greater_metric_to_watch_is_better):
            # STORE THE CURRENT metric AS BEST
            self.best_metric = metric
            self.sg_logger.add_checkpoint(tag=self.ckpt_best_name, state_dict=state, global_step=epoch)

            # RUN PHASE CALLBACKS
            self.phase_callback_handler.on_validation_end_best_epoch(context)

            if isinstance(metric, torch.Tensor):
                metric = metric.item()
            logger.info("Best checkpoint overriden: validation " + self.metric_to_watch + ": " + str(metric))

        if self.training_params.average_best_models:
            net_for_averaging = self.ema_model.ema if self.ema else self.net
            state["net"] = self.model_weight_averaging.get_average_model(net_for_averaging, validation_results_tuple=validation_results_tuple)
            self.sg_logger.add_checkpoint(tag=self.average_model_checkpoint_filename, state_dict=state, global_step=epoch)

    def _prep_net_for_train(self) -> None:
        if self.arch_params is None:
            self._init_arch_params()

        # TODO: REMOVE THE BELOW LINE (FOR BACKWARD COMPATIBILITY)
        if self.checkpoint_params is None:
            self.checkpoint_params = HpmStruct(load_checkpoint=self.training_params.resume)

        self._net_to_device()

        # SET THE FLAG FOR DIFFERENT PARAMETER GROUP OPTIMIZER UPDATE
        self.update_param_groups = hasattr(self.net.module, "update_param_groups")

        self.checkpoint = {}
        self.strict_load = core_utils.get_param(self.training_params, "resume_strict_load", StrictLoad.ON)
        self.load_ema_as_net = False
        self.load_checkpoint = core_utils.get_param(self.training_params, "resume", False)
        self.external_checkpoint_path = core_utils.get_param(self.training_params, "resume_path")
        self.ckpt_name = core_utils.get_param(self.training_params, "ckpt_name", "ckpt_latest.pth")
        self.resume_from_remote_sg_logger = core_utils.get_param(self.training_params, "resume_from_remote_sg_logger", False)
        self.load_checkpoint = self.load_checkpoint or self.external_checkpoint_path is not None or self.resume_from_remote_sg_logger

    def _init_arch_params(self) -> None:
        default_arch_params = HpmStruct()
        arch_params = getattr(self.net, "arch_params", default_arch_params)
        self.arch_params = default_arch_params
        if arch_params is not None:
            self.arch_params.override(**arch_params.to_dict())

    # FIXME - we need to resolve flake8's 'function is too complex' for this function
    def train(
        self,
        model: nn.Module,
        training_params: dict = None,
        train_loader: DataLoader = None,
        valid_loader: DataLoader = None,
        additional_configs_to_log: Dict = None,
    ):  # noqa: C901
        """

        train - Trains the Model

        IMPORTANT NOTE: Additional batch parameters can be added as a third item (optional) if a tuple is returned by
          the data loaders, as dictionary. The phase context will hold the additional items, under an attribute with
          the same name as the key in this dictionary. Then such items can be accessed through phase callbacks.

            :param additional_configs_to_log: Dict, dictionary containing configs that will be added to the training's
                sg_logger. Format should be {"Config_title_1": {...}, "Config_title_2":{..}}.
            :param model: torch.nn.Module, model to train.

            :param train_loader: Dataloader for train set.
            :param valid_loader: Dataloader for validation.
            :param training_params:

                - `resume` : bool (default=False)

                    Whether to continue training from ckpt with the same experiment name
                     (i.e resume from CKPT_ROOT_DIR/EXPERIMENT_NAME/CKPT_NAME)

                - `ckpt_name` : str (default=ckpt_latest.pth)

                    The checkpoint (.pth file) filename in CKPT_ROOT_DIR/EXPERIMENT_NAME/ to use when resume=True and
                     resume_path=None

                - `resume_path`: str (default=None)

                    Explicit checkpoint path (.pth file) to use to resume training.

                - `max_epochs` : int

                    Number of epochs to run training.

                - `lr_updates` : list(int)

                    List of fixed epoch numbers to perform learning rate updates when `lr_mode='step'`.

                - `lr_decay_factor` : float

                    Decay factor to apply to the learning rate at each update when `lr_mode='step'`.


                -  `lr_mode` : str

                    Learning rate scheduling policy, one of ['step','poly','cosine','function'].

                    'step' refers to constant updates at epoch numbers passed through `lr_updates`. Each update decays the learning rate by `lr_decay_factor`.

                    'cosine' refers to the Cosine Anealing policy as mentioned in https://arxiv.org/abs/1608.03983.
                      The final learning rate ratio is controlled by `cosine_final_lr_ratio` training parameter.

                    'poly' refers to the polynomial decrease: in each epoch iteration `self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)), 0.9)`

                    'function' refers to a user-defined learning rate scheduling function, that is passed through `lr_schedule_function`.

                - `lr_schedule_function` : Union[callable,None]

                    Learning rate scheduling function to be used when `lr_mode` is 'function'.

                - `warmup_mode`: Union[str, Type[LRCallbackBase], None]

                    If not None, define how the learning rate will be increased during the warmup phase.
                    Currently, only 'warmup_linear_epoch' and `warmup_linear_step` modes are supported.

                - `lr_warmup_epochs` : int (default=0)

                    Number of epochs for learning rate warm up - see https://arxiv.org/pdf/1706.02677.pdf (Section 2.2).
                    Relevant for `warmup_mode=warmup_linear_epoch`.
                    When lr_warmup_epochs > 0, the learning rate will be increased linearly from 0 to the `initial_lr`
                    once per epoch.

                - `lr_warmup_steps` : int (default=0)

                    Number of steps for learning rate warm up - see https://arxiv.org/pdf/1706.02677.pdf (Section 2.2).
                    Relevant for `warmup_mode=warmup_linear_step`.
                    When lr_warmup_steps > 0, the learning rate will be increased linearly from 0 to the `initial_lr`
                    for a total number of steps according to formula: min(lr_warmup_steps, len(train_loader)).
                    The capping is done to avoid interference of warmup with epoch-based schedulers.

                - `cosine_final_lr_ratio` : float (default=0.01)
                    Final learning rate ratio (only relevant when `lr_mode`='cosine'). The cosine starts from initial_lr and reaches
                     initial_lr * cosine_final_lr_ratio in last epoch

                - `inital_lr` : float

                    Initial learning rate.

                - `loss` : Union[nn.module, str]

                    Loss function for training.
                    One of SuperGradient's built in options:

                              "cross_entropy": LabelSmoothingCrossEntropyLoss,
                              "mse": MSELoss,
                              "r_squared_loss": RSquaredLoss,
                              "detection_loss": YoLoV3DetectionLoss,
                              "shelfnet_ohem_loss": ShelfNetOHEMLoss,
                              "shelfnet_se_loss": ShelfNetSemanticEncodingLoss,
                              "ssd_loss": SSDLoss,


                    or user defined nn.module loss function.

                    IMPORTANT: forward(...) should return a (loss, loss_items) tuple where loss is the tensor used
                    for backprop (i.e what your original loss function returns), and loss_items should be a tensor of
                    shape (n_items), of values computed during the forward pass which we desire to log over the
                    entire epoch. For example- the loss itself should always be logged. Another example is a scenario
                    where the computed loss is the sum of a few components we would like to log- these entries in
                    loss_items).

                    IMPORTANT:When dealing with external loss classes, to logg/monitor the loss_items as described
                    above by specific string name:

                    Set a "component_names" property in the loss class, whos instance is passed through train_params,
                     to be a list of strings, of length n_items who's ith element is the name of the ith entry in loss_items.
                     Then each item will be logged, rendered on tensorboard and "watched" (i.e saving model checkpoints
                     according to it) under <LOSS_CLASS.__name__>"/"<COMPONENT_NAME>. If a single item is returned rather then a
                     tuple, it would be logged under <LOSS_CLASS.__name__>. When there is no such attributed, the items
                     will be named <LOSS_CLASS.__name__>"/"Loss_"<IDX> according to the length of loss_items

                    For example:
                        class MyLoss(_Loss):
                            ...
                            def forward(self, inputs, targets):
                                ...
                                total_loss = comp1 + comp2
                                loss_items = torch.cat((total_loss.unsqueeze(0),comp1.unsqueeze(0), comp2.unsqueeze(0)).detach()
                                return total_loss, loss_items
                            ...
                            @property
                            def component_names(self):
                                return ["total_loss", "my_1st_component", "my_2nd_component"]

                    Trainer.train(...
                                    train_params={"loss":MyLoss(),
                                                    ...
                                                    "metric_to_watch": "MyLoss/my_1st_component"}

                        This will write to log and monitor MyLoss/total_loss, MyLoss/my_1st_component,
                         MyLoss/my_2nd_component.

                   For example:
                        class MyLoss2(_Loss):
                            ...
                            def forward(self, inputs, targets):
                                ...
                                total_loss = comp1 + comp2
                                loss_items = torch.cat((total_loss.unsqueeze(0),comp1.unsqueeze(0), comp2.unsqueeze(0)).detach()
                                return total_loss, loss_items
                            ...

                    Trainer.train(...
                                    train_params={"loss":MyLoss(),
                                                    ...
                                                    "metric_to_watch": "MyLoss2/loss_0"}

                        This will write to log and monitor MyLoss2/loss_0, MyLoss2/loss_1, MyLoss2/loss_2
                        as they have been named by their positional index in loss_items.

                    Since running logs will save the loss_items in some internal state, it is recommended that
                    loss_items are detached from their computational graph for memory efficiency.

                - `optimizer` : Union[str, torch.optim.Optimizer]

                    Optimization algorithm. One of ['Adam','SGD','RMSProp'] corresponding to the torch.optim
                    optimzers implementations, or any object that implements torch.optim.Optimizer.

                - `criterion_params` : dict

                    Loss function parameters.

                - `optimizer_params` : dict
                    When `optimizer` is one of ['Adam','SGD','RMSProp'], it will be initialized with optimizer_params.

                    (see https://pytorch.org/docs/stable/optim.html for the full list of
                    parameters for each optimizer).

                - `train_metrics_list` : list(torchmetrics.Metric)

                    Metrics to log during training. For more information on torchmetrics see
                    https://torchmetrics.rtfd.io/en/latest/.


                - `valid_metrics_list` : list(torchmetrics.Metric)

                    Metrics to log during validation/testing. For more information on torchmetrics see
                    https://torchmetrics.rtfd.io/en/latest/.


                - `loss_logging_items_names` : list(str)

                    The list of names/titles for the outputs returned from the loss functions forward pass (reminder-
                    the loss function should return the tuple (loss, loss_items)). These names will be used for
                    logging their values.

                - `metric_to_watch` : str (default="Accuracy")

                    will be the metric which the model checkpoint will be saved according to, and can be set to any
                    of the following:

                        a metric name (str) of one of the metric objects from the valid_metrics_list

                        a "metric_name" if some metric in valid_metrics_list has an attribute component_names which
                        is a list referring to the names of each entry in the output metric (torch tensor of size n)

                        one of "loss_logging_items_names" i.e which will correspond to an item returned during the
                        loss function's forward pass (see loss docs abov).

                    At the end of each epoch, if a new best metric_to_watch value is achieved, the models checkpoint
                    is saved in YOUR_PYTHON_PATH/checkpoints/ckpt_best.pth

                - `greater_metric_to_watch_is_better` : bool

                    When choosing a model's checkpoint to be saved, the best achieved model is the one that maximizes the
                     metric_to_watch when this parameter is set to True, and a one that minimizes it otherwise.

                - `ema` : bool (default=False)

                    Whether to use Model Exponential Moving Average (see
                    https://github.com/rwightman/pytorch-image-models ema implementation)

                - `batch_accumulate` : int (default=1)

                    Number of batches to accumulate before every backward pass.

                - `ema_params` : dict

                    Parameters for the ema model.

                - `zero_weight_decay_on_bias_and_bn` : bool (default=False)

                    Whether to apply weight decay on batch normalization parameters or not (ignored when the passed
                    optimizer has already been initialized).


                - `load_opt_params` : bool (default=True)

                    Whether to load the optimizers parameters as well when loading a model's checkpoint.

                - `run_validation_freq` : int (default=1)

                    The frequency in which validation is performed during training (i.e the validation is ran every
                     `run_validation_freq` epochs.

                - `save_model` : bool (default=True)

                    Whether to save the model checkpoints.

                - `silent_mode` : bool

                    Silents the print outs.

                - `mixed_precision` : bool

                    Whether to use mixed precision or not.

                - `save_ckpt_epoch_list` : list(int) (default=[])

                    List of fixed epoch indices the user wishes to save checkpoints in.

                - `average_best_models` : bool (default=False)

                    If set, a snapshot dictionary file and the average model will be saved / updated at every epoch
                    and evaluated only when training is completed. The snapshot file will only be deleted upon
                    completing the training. The snapshot dict will be managed on cpu.

                - `precise_bn` : bool (default=False)

                    Whether to use precise_bn calculation during the training.

                - `precise_bn_batch_size` : int (default=None)

                    The effective batch size we want to calculate the batchnorm on. For example, if we are training a model
                    on 8 gpus, with a batch of 128 on each gpu, a good rule of thumb would be to give it 8192
                    (ie: effective_batch_size * num_gpus = batch_per_gpu * num_gpus * num_gpus).
                    If precise_bn_batch_size is not provided in the training_params, the latter heuristic will be taken.

                - `seed` : int (default=42)

                    Random seed to be set for torch, numpy, and random. When using DDP each process will have it's seed
                    set to seed + rank.


                - `log_installed_packages` : bool (default=False)

                    When set, the list of all installed packages (and their versions) will be written to the tensorboard
                     and logfile (useful when trying to reproduce results).

                - `dataset_statistics` : bool (default=False)

                    Enable a statistic analysis of the dataset. If set to True the dataset will be analyzed and a report
                    will be added to the tensorboard along with some sample images from the dataset. Currently only
                    detection datasets are supported for analysis.

                -  `sg_logger` : Union[AbstractSGLogger, str] (defauls=base_sg_logger)

                    Define the SGLogger object for this training process. The SGLogger handles all disk writes, logs, TensorBoard, remote logging
                    and remote storage. By overriding the default base_sg_logger, you can change the storage location, support external monitoring and logging
                    or support remote storage.

                -   `sg_logger_params` : dict

                    SGLogger parameters

                -   `clip_grad_norm` : float

                    Defines a maximal L2 norm of the gradients. Values which exceed the given value will be clipped

                -   `lr_cooldown_epochs` : int (default=0)

                    Number of epochs to cooldown LR (i.e the last epoch from scheduling view point=max_epochs-cooldown).

                -   `pre_prediction_callback` : Callable (default=None)

                     When not None, this callback will be applied to images and targets, and returning them to be used
                      for the forward pass, and further computations. Args for this callable should be in the order
                      (inputs, targets, batch_idx) returning modified_inputs, modified_targets

                -   `ckpt_best_name` : str (default='ckpt_best.pth')

                    The best checkpoint (according to metric_to_watch) will be saved under this filename in the checkpoints directory.

                -   `max_train_batches`: int, for debug- when not None- will break out of inner train loop (i.e iterating over
                      train_loader) when reaching this number of batches. Usefull for debugging (default=None).

                -   `max_valid_batches`: int, for debug- when not None- will break out of inner valid loop (i.e iterating over
                      valid_loader) when reaching this number of batches. Usefull for debugging (default=None).

                -   `resume_from_remote_sg_logger`: bool (default=False),  bool (default=False), When true, ckpt_name (checkpoint filename
                       to resume i.e ckpt_latest.pth bydefault) will be downloaded into the experiment checkpoints directory
                       prior to loading weights, then training is resumed from that checkpoint. The source is unique to
                       every logger, and currently supported for WandB loggers only.

                       IMPORTANT: Only works for experiments that were ran with sg_logger_params.save_checkpoints_remote=True.
                       IMPORTANT: For WandB loggers, one must also pass the run id through the wandb_id arg in sg_logger_params.



        :return:
        """
        global logger
        if training_params is None:
            training_params = dict()

        self.train_loader = train_loader if train_loader is not None else self.train_loader
        self.valid_loader = valid_loader if valid_loader is not None else self.valid_loader

        if self.train_loader is None:
            raise ValueError("No `train_loader` found. Please provide a value for `train_loader`")

        if self.valid_loader is None:
            raise ValueError("No `valid_loader` found. Please provide a value for `valid_loader`")

        if hasattr(self.train_loader, "batch_sampler") and self.train_loader.batch_sampler is not None:
            batch_size = self.train_loader.batch_sampler.batch_size
        else:
            batch_size = self.train_loader.batch_size

        if len(self.train_loader.dataset) % batch_size != 0 and not self.train_loader.drop_last:
            logger.warning("Train dataset size % batch_size != 0 and drop_last=False, this might result in smaller " "last batch.")
        self._set_dataset_params()

        if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            # Note: the dataloader uses sampler of the batch_sampler when it is not None.
            train_sampler = self.train_loader.batch_sampler.sampler if self.train_loader.batch_sampler is not None else self.train_loader.sampler
            if isinstance(train_sampler, SequentialSampler):
                raise ValueError(
                    "You are using a SequentialSampler on you training dataloader, while working on DDP. "
                    "This cancels the DDP benefits since it makes each process iterate through the entire dataset"
                )
            if not isinstance(train_sampler, (DistributedSampler, RepeatAugSampler)):
                logger.warning(
                    "The training sampler you are using might not support DDP. "
                    "If it doesnt, please use one of the following sampler: DistributedSampler, RepeatAugSampler"
                )
        self.training_params = TrainingParams()
        self.training_params.override(**training_params)

        self.net = model
        self._prep_net_for_train()
        if not self.ddp_silent_mode:
            self._initialize_sg_logger_objects(additional_configs_to_log)
        self._load_checkpoint_to_model()

        # SET RANDOM SEED
        random_seed(is_ddp=device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, device=device_config.device, seed=self.training_params.seed)

        silent_mode = self.training_params.silent_mode or self.ddp_silent_mode
        # METRICS
        self._set_train_metrics(train_metrics_list=self.training_params.train_metrics_list)
        self._set_valid_metrics(valid_metrics_list=self.training_params.valid_metrics_list)

        # Store the metric to follow (loss\accuracy) and initialize as the worst value
        self.metric_to_watch = self.training_params.metric_to_watch
        self.greater_metric_to_watch_is_better = self.training_params.greater_metric_to_watch_is_better

        # Allowing loading instantiated loss or string
        if isinstance(self.training_params.loss, str):
            self.criterion = LossesFactory().get({self.training_params.loss: self.training_params.criterion_params})

        elif isinstance(self.training_params.loss, Mapping):
            self.criterion = LossesFactory().get(self.training_params.loss)

        elif isinstance(self.training_params.loss, nn.Module):
            self.criterion = self.training_params.loss

        self.criterion.to(device_config.device)

        self.max_epochs = self.training_params.max_epochs

        self.ema = self.training_params.ema

        self.precise_bn = self.training_params.precise_bn
        self.precise_bn_batch_size = self.training_params.precise_bn_batch_size

        self.batch_accumulate = self.training_params.batch_accumulate
        num_batches = len(self.train_loader)

        if self.ema:
            self.ema_model = self._instantiate_ema_model(self.training_params.ema_params)
            self.ema_model.updates = self.start_epoch * num_batches // self.batch_accumulate
            if self.load_checkpoint:
                if "ema_net" in self.checkpoint.keys():
                    self.ema_model.ema.load_state_dict(self.checkpoint["ema_net"])
                else:
                    self.ema = False
                    logger.warning("[Warning] Checkpoint does not include EMA weights, continuing training without EMA.")

        self.run_validation_freq = self.training_params.run_validation_freq
        validation_results_tuple = (0, 0)
        inf_time = 0
        timer = core_utils.Timer(device_config.device)

        # IF THE LR MODE IS NOT DEFAULT TAKE IT FROM THE TRAINING PARAMS
        self.lr_mode = self.training_params.lr_mode
        load_opt_params = self.training_params.load_opt_params

        self.phase_callbacks = self.training_params.phase_callbacks or []
        self.phase_callbacks = ListFactory(CallbacksFactory()).get(self.phase_callbacks)

        if self.lr_mode is not None:
            sg_lr_callback_cls = LR_SCHEDULERS_CLS_DICT[self.lr_mode]
            self.phase_callbacks.append(
                sg_lr_callback_cls(
                    train_loader_len=len(self.train_loader),
                    net=self.net,
                    training_params=self.training_params,
                    update_param_groups=self.update_param_groups,
                    **self.training_params.to_dict(),
                )
            )

        warmup_mode = self.training_params.warmup_mode
        warmup_callback_cls = None
        if isinstance(warmup_mode, str):
            warmup_callback_cls = LR_WARMUP_CLS_DICT[warmup_mode]
        elif isinstance(warmup_mode, type) and issubclass(warmup_mode, LRCallbackBase):
            warmup_callback_cls = warmup_mode
        elif warmup_mode is not None:
            pass
        else:
            raise RuntimeError("warmup_mode has to be either a name of a mode (str) or a subclass of PhaseCallback")

        if warmup_callback_cls is not None:
            self.phase_callbacks.append(
                warmup_callback_cls(
                    train_loader_len=len(self.train_loader),
                    net=self.net,
                    training_params=self.training_params,
                    update_param_groups=self.update_param_groups,
                    **self.training_params.to_dict(),
                )
            )

        self._add_metrics_update_callback(Phase.TRAIN_BATCH_END)
        self._add_metrics_update_callback(Phase.VALIDATION_BATCH_END)

        self.phase_callback_handler = CallbackHandler(callbacks=self.phase_callbacks)

        if not self.ddp_silent_mode:
            if self.training_params.dataset_statistics:
                dataset_statistics_logger = DatasetStatisticsTensorboardLogger(self.sg_logger)
                dataset_statistics_logger.analyze(self.train_loader, all_classes=self.classes, title="Train-set", anchors=self.net.module.arch_params.anchors)
                dataset_statistics_logger.analyze(self.valid_loader, all_classes=self.classes, title="val-set")

        sg_trainer_utils.log_uncaught_exceptions(logger)

        if not self.load_checkpoint or self.load_weights_only:
            # WHEN STARTING TRAINING FROM SCRATCH, DO NOT LOAD OPTIMIZER PARAMS (EVEN IF LOADING BACKBONE)
            self.start_epoch = 0
            self._reset_best_metric()
            load_opt_params = False

        if isinstance(self.training_params.optimizer, str) or (
            inspect.isclass(self.training_params.optimizer) and issubclass(self.training_params.optimizer, torch.optim.Optimizer)
        ):
            self.optimizer = build_optimizer(net=self.net, lr=self.training_params.initial_lr, training_params=self.training_params)
        elif isinstance(self.training_params.optimizer, torch.optim.Optimizer):
            self.optimizer = self.training_params.optimizer
        else:
            raise UnsupportedOptimizerFormat()

        # VERIFY GRADIENT CLIPPING VALUE
        if self.training_params.clip_grad_norm is not None and self.training_params.clip_grad_norm <= 0:
            raise TypeError("Params", "Invalid clip_grad_norm")

        if self.load_checkpoint and load_opt_params:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])

        self.pre_prediction_callback = CallbacksFactory().get(self.training_params.pre_prediction_callback)

        self._initialize_mixed_precision(self.training_params.mixed_precision)

        self.ckpt_best_name = self.training_params.ckpt_best_name

        if self.training_params.max_train_batches is not None:
            if self.training_params.max_train_batches > len(self.train_loader):
                logger.warning("max_train_batches is greater than len(self.train_loader) and will have no effect.")
            elif self.training_params.max_train_batches <= 0:
                raise ValueError("max_train_batches must be positive.")

        if self.training_params.max_valid_batches is not None:
            if self.training_params.max_valid_batches > len(self.valid_loader):
                logger.warning("max_valid_batches is greater than len(self.valid_loader) and will have no effect.")
            elif self.training_params.max_valid_batches <= 0:
                raise ValueError("max_valid_batches must be positive.")

        self.max_train_batches = self.training_params.max_train_batches
        self.max_valid_batches = self.training_params.max_valid_batches

        # STATE ATTRIBUTE SET HERE FOR SUBSEQUENT TRAIN() CALLS
        self._first_backward = True

        context = PhaseContext(
            optimizer=self.optimizer,
            net=self.net,
            experiment_name=self.experiment_name,
            ckpt_dir=self.checkpoints_dir_path,
            criterion=self.criterion,
            lr_warmup_epochs=self.training_params.lr_warmup_epochs,
            sg_logger=self.sg_logger,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            training_params=self.training_params,
            ddp_silent_mode=self.ddp_silent_mode,
            checkpoint_params=self.checkpoint_params,
            architecture=self.architecture,
            arch_params=self.arch_params,
            metric_to_watch=self.metric_to_watch,
            device=device_config.device,
            context_methods=self._get_context_methods(Phase.PRE_TRAINING),
            ema_model=self.ema_model,
        )
        self.phase_callback_handler.on_training_start(context)

        first_batch = next(iter(self.train_loader))
        inputs, _, _ = sg_trainer_utils.unpack_batch_items(first_batch)

        log_main_training_params(
            multi_gpu=device_config.multi_gpu,
            num_gpus=get_world_size(),
            batch_size=len(inputs),
            batch_accumulate=self.batch_accumulate,
            train_dataset_length=len(self.train_loader.dataset),
            train_dataloader_len=len(self.train_loader),
        )

        self._set_net_preprocessing_from_valid_loader()
        try:
            # HEADERS OF THE TRAINING PROGRESS
            if not silent_mode:
                logger.info(f"Started training for {self.max_epochs - self.start_epoch} epochs ({self.start_epoch}/" f"{self.max_epochs - 1})\n")
            for epoch in range(self.start_epoch, self.max_epochs):
                if context.stop_training:
                    logger.info("Request to stop training has been received, stopping training")
                    break

                # Phase.TRAIN_EPOCH_START
                # RUN PHASE CALLBACKS
                context.update_context(epoch=epoch)
                self.phase_callback_handler.on_train_loader_start(context)

                # IN DDP- SET_EPOCH WILL CAUSE EVERY PROCESS TO BE EXPOSED TO THE ENTIRE DATASET BY SHUFFLING WITH A
                # DIFFERENT SEED EACH EPOCH START
                if (
                    device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL
                    and hasattr(self.train_loader, "sampler")
                    and hasattr(self.train_loader.sampler, "set_epoch")
                ):
                    self.train_loader.sampler.set_epoch(epoch)

                train_metrics_tuple = self._train_epoch(epoch=epoch, silent_mode=silent_mode)

                # Phase.TRAIN_EPOCH_END
                # RUN PHASE CALLBACKS
                train_metrics_dict = get_metrics_dict(train_metrics_tuple, self.train_metrics, self.loss_logging_items_names)

                context.update_context(metrics_dict=train_metrics_dict)
                self.phase_callback_handler.on_train_loader_end(context)

                # CALCULATE PRECISE BATCHNORM STATS
                if self.precise_bn:
                    compute_precise_bn_stats(
                        model=self.net, loader=self.train_loader, precise_bn_batch_size=self.precise_bn_batch_size, num_gpus=get_world_size()
                    )
                    if self.ema:
                        compute_precise_bn_stats(
                            model=self.ema_model.ema,
                            loader=self.train_loader,
                            precise_bn_batch_size=self.precise_bn_batch_size,
                            num_gpus=get_world_size(),
                        )

                # model switch - we replace self.net.module with the ema model for the testing and saving part
                # and then switch it back before the next training epoch
                if self.ema:
                    self.ema_model.update_attr(self.net)
                    keep_model = self.net
                    self.net = self.ema_model.ema

                # RUN TEST ON VALIDATION SET EVERY self.run_validation_freq EPOCHS
                if (epoch + 1) % self.run_validation_freq == 0:
                    self.phase_callback_handler.on_validation_loader_start(context)
                    timer.start()
                    validation_results_tuple = self._validate_epoch(epoch=epoch, silent_mode=silent_mode)
                    inf_time = timer.stop()

                    # Phase.VALIDATION_EPOCH_END
                    # RUN PHASE CALLBACKS
                    valid_metrics_dict = get_metrics_dict(validation_results_tuple, self.valid_metrics, self.loss_logging_items_names)

                    context.update_context(metrics_dict=valid_metrics_dict)
                    self.phase_callback_handler.on_validation_loader_end(context)

                if self.ema:
                    self.net = keep_model

                if not self.ddp_silent_mode:
                    # SAVING AND LOGGING OCCURS ONLY IN THE MAIN PROCESS (IN CASES THERE ARE SEVERAL PROCESSES - DDP)
                    self._write_to_disk_operations(
                        train_metrics=train_metrics_tuple,
                        validation_results=validation_results_tuple,
                        lr_dict=self._epoch_start_logging_values,
                        inf_time=inf_time,
                        epoch=epoch,
                        context=context,
                    )
                    self.sg_logger.upload()

            # Evaluating the average model and removing snapshot averaging file if training is completed
            if self.training_params.average_best_models:
                self._validate_final_average_model(cleanup_snapshots_pkl_file=True)

        except KeyboardInterrupt:
            logger.info(
                "\n[MODEL TRAINING EXECUTION HAS BEEN INTERRUPTED]... Please wait until SOFT-TERMINATION process "
                "finishes and saves all of the Model Checkpoints and log files before terminating..."
            )
            logger.info("For HARD Termination - Stop the process again")

        finally:
            if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
                # CLEAN UP THE MULTI-GPU PROCESS GROUP WHEN DONE
                if torch.distributed.is_initialized() and self.training_params.kill_ddp_pgroup_on_end:
                    torch.distributed.destroy_process_group()

            # PHASE.TRAIN_END
            self.phase_callback_handler.on_training_end(context)

            if not self.ddp_silent_mode:
                self.sg_logger.close()

    def _set_net_preprocessing_from_valid_loader(self):
        if isinstance(self.net.module, HasPredict) and isinstance(self.valid_loader.dataset, HasPreprocessingParams):
            try:
                self.net.module.set_dataset_processing_params(**self.valid_loader.dataset.get_dataset_preprocessing_params())
            except Exception as e:
                logger.warning(
                    f"Could not set preprocessing pipeline from the validation dataset:\n {e}.\n Before calling"
                    "predict make sure to call set_dataset_processing_params."
                )

    def _reset_best_metric(self):
        self.best_metric = -1 * np.inf if self.greater_metric_to_watch_is_better else np.inf

    def _reset_metrics(self):
        for metric in ("train_metrics", "valid_metrics", "test_metrics"):
            if hasattr(self, metric) and getattr(self, metric) is not None:
                getattr(self, metric).reset()

    @resolve_param("train_metrics_list", ListFactory(MetricsFactory()))
    def _set_train_metrics(self, train_metrics_list):
        self.train_metrics = MetricCollection(train_metrics_list)

        for metric_name, metric in self.train_metrics.items():
            if hasattr(metric, "greater_component_is_better"):
                self.greater_train_metrics_is_better.update(metric.greater_component_is_better)
            elif hasattr(metric, "greater_is_better"):
                self.greater_train_metrics_is_better[metric_name] = metric.greater_is_better
            else:
                self.greater_train_metrics_is_better[metric_name] = None

    @resolve_param("valid_metrics_list", ListFactory(MetricsFactory()))
    def _set_valid_metrics(self, valid_metrics_list):
        self.valid_metrics = MetricCollection(valid_metrics_list)

        for metric_name, metric in self.valid_metrics.items():
            if hasattr(metric, "greater_component_is_better"):
                self.greater_valid_metrics_is_better.update(metric.greater_component_is_better)
            elif hasattr(metric, "greater_is_better"):
                self.greater_valid_metrics_is_better[metric_name] = metric.greater_is_better
            else:
                self.greater_valid_metrics_is_better[metric_name] = None

    @resolve_param("test_metrics_list", ListFactory(MetricsFactory()))
    def _set_test_metrics(self, test_metrics_list):
        self.test_metrics = MetricCollection(test_metrics_list)

    def _initialize_mixed_precision(self, mixed_precision_enabled: bool):
        # SCALER IS ALWAYS INITIALIZED BUT IS DISABLED IF MIXED PRECISION WAS NOT SET
        self.scaler = GradScaler(enabled=mixed_precision_enabled)

        if mixed_precision_enabled:
            assert device_config.device.startswith("cuda"), "mixed precision is not available for CPU"
            if device_config.multi_gpu == MultiGPUMode.DATA_PARALLEL:
                # IN DATAPARALLEL MODE WE NEED TO WRAP THE FORWARD FUNCTION OF OUR MODEL SO IT WILL RUN WITH AUTOCAST.
                # BUT SINCE THE MODULE IS CLONED TO THE DEVICES ON EACH FORWARD CALL OF A DATAPARALLEL MODEL,
                # WE HAVE TO REGISTER THE WRAPPER BEFORE EVERY FORWARD CALL
                def hook(module, _):
                    module.forward = MultiGPUModeAutocastWrapper(module.forward)

                self.net.module.register_forward_pre_hook(hook=hook)

            if self.load_checkpoint:
                scaler_state_dict = core_utils.get_param(self.checkpoint, "scaler_state_dict")
                if scaler_state_dict is None:
                    logger.warning("Mixed Precision - scaler state_dict not found in loaded model. This may case issues " "with loss scaling")
                else:
                    self.scaler.load_state_dict(scaler_state_dict)

    def _validate_final_average_model(self, cleanup_snapshots_pkl_file=False):
        """
        Testing the averaged model by loading the last saved average checkpoint and running test.
        Will be loaded to each of DDP processes
        :param cleanup_pkl_file: a flag for deleting the 10 best snapshots dictionary
        """
        logger.info("RUNNING ADDITIONAL TEST ON THE AVERAGED MODEL...")

        keep_state_dict = deepcopy(self.net.state_dict())
        # SETTING STATE DICT TO THE AVERAGE MODEL FOR EVALUATION
        average_model_ckpt_path = os.path.join(self.checkpoints_dir_path, self.average_model_checkpoint_filename)
        local_rank = get_local_rank()

        # WAIT FOR MASTER RANK TO SAVE THE CKPT BEFORE WE TRY TO READ IT.
        with wait_for_the_master(local_rank):
            average_model_sd = read_ckpt_state_dict(average_model_ckpt_path)["net"]

        self.net.load_state_dict(average_model_sd)
        # testing the averaged model and save instead of best model if needed
        averaged_model_results_tuple = self._validate_epoch(epoch=self.max_epochs)

        # Reverting the current model
        self.net.load_state_dict(keep_state_dict)

        if not self.ddp_silent_mode:
            average_model_tb_titles = ["Averaged Model " + x for x in self.results_titles[-1 * len(averaged_model_results_tuple) :]]
            write_struct = ""
            for ind, title in enumerate(average_model_tb_titles):
                write_struct += "%s: %.3f  \n  " % (title, averaged_model_results_tuple[ind])
                self.sg_logger.add_scalar(title, averaged_model_results_tuple[ind], global_step=self.max_epochs)

            self.sg_logger.add_text("Averaged_Model_Performance", write_struct, self.max_epochs)
            if cleanup_snapshots_pkl_file:
                self.model_weight_averaging.cleanup()

    @property
    def get_arch_params(self):
        return self.arch_params.to_dict()

    @property
    def get_structure(self):
        return self.net.module.structure

    @property
    def get_architecture(self):
        return self.architecture

    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name

    def _re_build_model(self, arch_params={}):
        """
        arch_params : dict
            Architecture H.P. e.g.: block, num_blocks, num_classes, etc.
        :return:
        """
        if "num_classes" not in arch_params.keys():
            if self.dataset_interface is None:
                raise Exception("Error", "Number of classes not defined in arch params and dataset is not defined")
            else:
                arch_params["num_classes"] = len(self.classes)

        self.arch_params = core_utils.HpmStruct(**arch_params)
        self.classes = self.arch_params.num_classes
        self.net = self._instantiate_net(self.architecture, self.arch_params, self.checkpoint_params)
        # save the architecture for neural architecture search
        if hasattr(self.net, "structure"):
            self.architecture = self.net.structure

        self.net.to(device_config.device)

        if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            logger.warning("Warning: distributed training is not supported in re_build_model()")
        self.net = torch.nn.DataParallel(self.net, device_ids=get_device_ids()) if device_config.multi_gpu else core_utils.WrappedModel(self.net)

    @property
    def get_module(self):
        return self.net

    def set_module(self, module):
        self.net = module

    def _switch_device(self, new_device):
        device_config.device = new_device
        self.net.to(device_config.device)

    # FIXME - we need to resolve flake8's 'function is too complex' for this function
    def _load_checkpoint_to_model(self):  # noqa: C901 - too complex
        """
        Copies the source checkpoint to a local folder and loads the checkpoint's data to the model using the
         attributes:

         strict:           See StrictLoad class documentation for details.
         load_backbone:    loads the provided checkpoint to self.net.backbone instead of self.net

        NOTE: 'acc', 'epoch', 'optimizer_state_dict' and the logs are NOT loaded if self.zeroize_prev_train_params
         is True
        """
        with wait_for_the_master(get_local_rank()):
            if self.resume_from_remote_sg_logger and not self.ddp_silent_mode:
                self.sg_logger.download_remote_ckpt(ckpt_name=self.ckpt_name)

        if self.load_checkpoint or self.external_checkpoint_path:
            # GET LOCAL PATH TO THE CHECKPOINT FILE FIRST
            ckpt_root_dir = str(Path(self.checkpoints_dir_path).parent)
            ckpt_local_path = get_ckpt_local_path(
                ckpt_root_dir=ckpt_root_dir,
                experiment_name=self.experiment_name,
                ckpt_name=self.ckpt_name,
                external_checkpoint_path=self.external_checkpoint_path,
            )

            # LOAD CHECKPOINT TO MODEL
            self.checkpoint = load_checkpoint_to_model(
                ckpt_local_path=ckpt_local_path,
                load_backbone=self.load_backbone,
                net=self.net,
                strict=self.strict_load.value if isinstance(self.strict_load, StrictLoad) else self.strict_load,
                load_weights_only=self.load_weights_only,
                load_ema_as_net=self.load_ema_as_net,
            )

            if "ema_net" in self.checkpoint.keys():
                logger.warning(
                    "[WARNING] Main network has been loaded from checkpoint but EMA network exists as "
                    "well. It "
                    " will only be loaded during validation when training with ema=True. "
                )

        # UPDATE TRAINING PARAMS IF THEY EXIST & WE ARE NOT LOADING AN EXTERNAL MODEL's WEIGHTS
        self.best_metric = self.checkpoint["acc"] if "acc" in self.checkpoint.keys() else -1
        self.start_epoch = self.checkpoint["epoch"] if "epoch" in self.checkpoint.keys() else 0

    def _prep_for_test(
        self, test_loader: torch.utils.data.DataLoader = None, loss=None, test_metrics_list=None, loss_logging_items_names=None, test_phase_callbacks=None
    ):
        """Run commands that are common to all models"""
        # SET THE MODEL IN evaluation STATE
        self.net.eval()

        # IF SPECIFIED IN THE FUNCTION CALL - OVERRIDE THE self ARGUMENTS
        self.test_loader = test_loader or self.test_loader
        self.criterion = loss or self.criterion
        self.loss_logging_items_names = loss_logging_items_names or self.loss_logging_items_names
        self.phase_callbacks = test_phase_callbacks or self.phase_callbacks

        if self.phase_callbacks is None:
            self.phase_callbacks = []

        if test_metrics_list:
            self._set_test_metrics(test_metrics_list)
            self._add_metrics_update_callback(Phase.TEST_BATCH_END)
            self.phase_callback_handler = CallbackHandler(self.phase_callbacks)

        # WHEN TESTING WITHOUT A LOSS FUNCTION- CREATE EPOCH HEADERS FOR PRINTS
        if self.criterion is None:
            self.loss_logging_items_names = []

        if self.test_metrics is None:
            raise ValueError(
                "Metrics are required to perform test. Pass them through test_metrics_list arg when "
                "calling test or through training_params when calling train(...)"
            )
        if self.test_loader is None:
            raise ValueError("Test dataloader is required to perform test. Make sure to either pass it through " "test_loader arg.")

        # RESET METRIC RUNNERS
        self._reset_metrics()
        self.test_metrics.to(device_config.device)

        if self.arch_params is None:
            self._init_arch_params()
        self._net_to_device()

    def _add_metrics_update_callback(self, phase: Phase):
        """
        Adds MetricsUpdateCallback to be fired at phase

        :param phase: Phase for the metrics callback to be fired at
        """
        self.phase_callbacks.append(MetricsUpdateCallback(phase))

    def _initialize_sg_logger_objects(self, additional_configs_to_log: Dict = None):
        """Initialize object that collect, write to disk, monitor and store remotely all training outputs"""
        sg_logger = core_utils.get_param(self.training_params, "sg_logger")

        # OVERRIDE SOME PARAMETERS TO MAKE SURE THEY MATCH THE TRAINING PARAMETERS
        general_sg_logger_params = {
            "experiment_name": self.experiment_name,
            "storage_location": "local",
            "resumed": self.load_checkpoint,
            "training_params": self.training_params,
            "checkpoints_dir_path": self.checkpoints_dir_path,
        }

        if sg_logger is None:
            raise RuntimeError("sg_logger must be defined in training params (see default_training_params)")

        if isinstance(sg_logger, AbstractSGLogger):
            self.sg_logger = sg_logger
        elif isinstance(sg_logger, str):

            sg_logger_cls = SG_LOGGERS.get(sg_logger)
            if sg_logger_cls is None:
                raise RuntimeError(f"sg_logger={sg_logger} not registered in SuperGradients. Available {list(SG_LOGGERS.keys())}")

            sg_logger_params = core_utils.get_param(self.training_params, "sg_logger_params", {})
            if issubclass(sg_logger_cls, BaseSGLogger):
                sg_logger_params = {**sg_logger_params, **general_sg_logger_params}

            # Some sg_logger require model_name, but not all of them.
            if "model_name" in get_callable_param_names(sg_logger_cls.__init__):
                if sg_logger_params.get("model_name") is None:
                    # Use the model name used in `models.get(...)` if relevant
                    sg_logger_params["model_name"] = get_model_name(self.net.module)

                if sg_logger_params["model_name"] is None:
                    raise ValueError(
                        f'`model_name` is required to use `training_hyperparams.sg_logger="{sg_logger}"`.\n'
                        'Please set `training_hyperparams.sg_logger_params.model_name="<your-model-name>"`.\n'
                        "Note that specifying `model_name` is not required when the model was loaded using `models.get(...)`."
                    )

            self.sg_logger = sg_logger_cls(**sg_logger_params)
        else:
            raise RuntimeError("sg_logger can be either an sg_logger name (str) or an instance of AbstractSGLogger")

        if not isinstance(self.sg_logger, BaseSGLogger):
            logger.warning(
                "WARNING! Using a user-defined sg_logger: files will not be automatically written to disk!\n"
                "Please make sure the provided sg_logger writes to disk or compose your sg_logger to BaseSGLogger"
            )

        # IN CASE SG_LOGGER UPDATED THE DIR PATH
        self.checkpoints_dir_path = self.sg_logger.local_dir()
        hyper_param_config = self._get_hyper_param_config()
        if additional_configs_to_log is not None:
            hyper_param_config["additional_configs_to_log"] = additional_configs_to_log
        self.sg_logger.add_config("hyper_params", hyper_param_config)
        self.sg_logger.flush()

    def _get_hyper_param_config(self):
        """
        Creates a training hyper param config for logging.
        """
        additional_log_items = {
            "initial_LR": self.training_params.initial_lr,
            "num_devices": get_world_size(),
            "multi_gpu": str(device_config.multi_gpu),
            "device_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        # ADD INSTALLED PACKAGE LIST + THEIR VERSIONS
        if self.training_params.log_installed_packages:
            pkg_list = list(map(lambda pkg: str(pkg), _get_installed_distributions()))
            additional_log_items["installed_packages"] = pkg_list
        hyper_param_config = {
            "arch_params": self.arch_params.__dict__,
            "checkpoint_params": self.checkpoint_params.__dict__,
            "training_hyperparams": self.training_params.__dict__,
            "dataset_params": self.dataset_params.__dict__,
            "additional_log_items": additional_log_items,
        }
        return hyper_param_config

    def _write_to_disk_operations(self, train_metrics: tuple, validation_results: tuple, lr_dict: dict, inf_time: float, epoch: int, context: PhaseContext):
        """Run the various logging operations, e.g.: log file, Tensorboard, save checkpoint etc."""
        # STORE VALUES IN A TENSORBOARD FILE
        train_results = list(train_metrics) + list(validation_results) + [inf_time]
        all_titles = self.results_titles + ["Inference Time"]

        result_dict = {all_titles[i]: train_results[i] for i in range(len(train_results))}
        self.sg_logger.add_scalars(tag_scalar_dict=result_dict, global_step=epoch)
        self.sg_logger.add_scalars(tag_scalar_dict=lr_dict, global_step=epoch)

        # SAVE THE CHECKPOINT
        if self.training_params.save_model:
            self._save_checkpoint(self.optimizer, epoch + 1, validation_results, context)

    def _get_epoch_start_logging_values(self) -> dict:
        """Get all the values that should be logged at the start of each epoch.
        This is useful for values like Learning Rate that can change over an epoch."""
        lrs = [self.optimizer.param_groups[i]["lr"] for i in range(len(self.optimizer.param_groups))]
        lr_titles = ["LR/Param_group_" + str(i) for i in range(len(self.optimizer.param_groups))] if len(self.optimizer.param_groups) > 1 else ["LR"]
        lr_dict = {lr_titles[i]: lrs[i] for i in range(len(lrs))}
        return lr_dict

    def test(
        self,
        model: nn.Module = None,
        test_loader: torch.utils.data.DataLoader = None,
        loss: torch.nn.modules.loss._Loss = None,
        silent_mode: bool = False,
        test_metrics_list=None,
        loss_logging_items_names=None,
        metrics_progress_verbose=False,
        test_phase_callbacks=None,
        use_ema_net=True,
    ) -> tuple:
        """
        Evaluates the model on given dataloader and metrics.
        :param model: model to perfrom test on. When none is given, will try to use self.net (defalut=None).
        :param test_loader: dataloader to perform test on.
        :param test_metrics_list: (list(torchmetrics.Metric)) metrics list for evaluation.
        :param silent_mode: (bool) controls verbosity
        :param metrics_progress_verbose: (bool) controls the verbosity of metrics progress (default=False). Slows down the program.
        :param use_ema_net (bool) whether to perform test on self.ema_model.ema (when self.ema_model.ema exists,
            otherwise self.net will be tested) (default=True)
        :return: results tuple (tuple) containing the loss items and metric values.

        All of the above args will override Trainer's corresponding attribute when not equal to None. Then evaluation
         is ran on self.test_loader with self.test_metrics.
        """

        self.net = model or self.net

        # IN CASE TRAINING WAS PERFROMED BEFORE TEST- MAKE SURE TO TEST THE EMA MODEL (UNLESS SPECIFIED OTHERWISE BY
        # use_ema_net)

        if use_ema_net and self.ema_model is not None:
            keep_model = self.net
            self.net = self.ema_model.ema

        self._prep_for_test(
            test_loader=test_loader,
            loss=loss,
            test_metrics_list=test_metrics_list,
            loss_logging_items_names=loss_logging_items_names,
            test_phase_callbacks=test_phase_callbacks,
        )

        context = PhaseContext(
            criterion=self.criterion,
            device=self.device,
            sg_logger=self.sg_logger,
            context_methods=self._get_context_methods(Phase.TEST_BATCH_END),
        )
        if test_metrics_list:
            context.update_context(test_metrics=self.test_metrics)

        self.phase_callback_handler.on_test_loader_start(context)
        test_results = self.evaluate(
            data_loader=self.test_loader,
            metrics=self.test_metrics,
            evaluation_type=EvaluationType.TEST,
            silent_mode=silent_mode,
            metrics_progress_verbose=metrics_progress_verbose,
        )
        self.phase_callback_handler.on_test_loader_end(context)

        # SWITCH BACK BETWEEN NETS SO AN ADDITIONAL TRAINING CAN BE DONE AFTER TEST
        if use_ema_net and self.ema_model is not None:
            self.net = keep_model

        self._first_backward = True

        test_results = get_metrics_dict(test_results, self.test_metrics, self.loss_logging_items_names)

        return test_results

    def _validate_epoch(self, epoch: int, silent_mode: bool = False) -> tuple:
        """
        Runs evaluation on self.valid_loader, with self.valid_metrics.

        :param epoch: (int) epoch idx
        :param silent_mode: (bool) controls verbosity

        :return: results tuple (tuple) containing the loss items and metric values.
        """

        self.net.eval()
        self._reset_metrics()
        self.valid_metrics.to(device_config.device)

        return self.evaluate(
            data_loader=self.valid_loader, metrics=self.valid_metrics, evaluation_type=EvaluationType.VALIDATION, epoch=epoch, silent_mode=silent_mode
        )

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        metrics: MetricCollection,
        evaluation_type: EvaluationType,
        epoch: int = None,
        silent_mode: bool = False,
        metrics_progress_verbose: bool = False,
    ):
        """
        Evaluates the model on given dataloader and metrics.

        :param data_loader: dataloader to perform evaluataion on
        :param metrics: (MetricCollection) metrics for evaluation
        :param evaluation_type: (EvaluationType) controls which phase callbacks will be used (for example, on batch end,
            when evaluation_type=EvaluationType.VALIDATION the Phase.VALIDATION_BATCH_END callbacks will be triggered)
        :param epoch: (int) epoch idx
        :param silent_mode: (bool) controls verbosity
        :param metrics_progress_verbose: (bool) controls the verbosity of metrics progress (default=False).
            Slows down the program significantly.

        :return: results tuple (tuple) containing the loss items and metric values.
        """

        # THE DISABLE FLAG CONTROLS WHETHER THE PROGRESS BAR IS SILENT OR PRINTS THE LOGS
        loss_avg_meter = core_utils.utils.AverageMeter()
        logging_values = None

        lr_warmup_epochs = self.training_params.lr_warmup_epochs if self.training_params else None
        context = PhaseContext(
            epoch=epoch,
            metrics_compute_fn=metrics,
            loss_avg_meter=loss_avg_meter,
            criterion=self.criterion,
            device=device_config.device,
            lr_warmup_epochs=lr_warmup_epochs,
            sg_logger=self.sg_logger,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            context_methods=self._get_context_methods(Phase.VALIDATION_BATCH_END),
        )

        with tqdm(data_loader, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True, disable=silent_mode) as progress_bar_data_loader:

            if not silent_mode:
                # PRINT TITLES
                pbar_start_msg = f"Validation epoch {epoch}" if evaluation_type == EvaluationType.VALIDATION else "Test"
                progress_bar_data_loader.set_description(pbar_start_msg)

            with torch.no_grad():
                for batch_idx, batch_items in enumerate(progress_bar_data_loader):
                    batch_items = core_utils.tensor_container_to_device(batch_items, device_config.device, non_blocking=True)
                    inputs, targets, additional_batch_items = sg_trainer_utils.unpack_batch_items(batch_items)

                    # TRIGGER PHASE CALLBACKS CORRESPONDING TO THE EVALUATION TYPE
                    context.update_context(batch_idx=batch_idx, inputs=inputs, target=targets, **additional_batch_items)
                    if evaluation_type == EvaluationType.VALIDATION:
                        self.phase_callback_handler.on_validation_batch_start(context)
                    else:
                        self.phase_callback_handler.on_test_batch_start(context)

                    output = self.net(inputs)
                    context.update_context(preds=output)

                    if self.criterion is not None:
                        # STORE THE loss_items ONLY, THE 1ST RETURNED VALUE IS THE loss FOR BACKPROP DURING TRAINING
                        loss_tuple = self._get_losses(output, targets)[1].cpu()
                        context.update_context(loss_log_items=loss_tuple)

                    # TRIGGER PHASE CALLBACKS CORRESPONDING TO THE EVALUATION TYPE
                    if evaluation_type == EvaluationType.VALIDATION:
                        self.phase_callback_handler.on_validation_batch_end(context)
                    else:
                        self.phase_callback_handler.on_test_batch_end(context)

                    # COMPUTE METRICS IF PROGRESS VERBOSITY IS SET
                    if metrics_progress_verbose and not silent_mode:
                        # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
                        logging_values = get_logging_values(loss_avg_meter, metrics, self.criterion)
                        pbar_message_dict = get_train_loop_description_dict(logging_values, metrics, self.loss_logging_items_names)

                        progress_bar_data_loader.set_postfix(**pbar_message_dict)

                    if evaluation_type == EvaluationType.VALIDATION and self.max_valid_batches is not None and self.max_valid_batches - 1 <= batch_idx:
                        break

            # NEED TO COMPUTE METRICS FOR THE FIRST TIME IF PROGRESS VERBOSITY IS NOT SET
            if not metrics_progress_verbose:
                # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
                logging_values = get_logging_values(loss_avg_meter, metrics, self.criterion)
                pbar_message_dict = get_train_loop_description_dict(logging_values, metrics, self.loss_logging_items_names)

                progress_bar_data_loader.set_postfix(**pbar_message_dict)

            # TODO: SUPPORT PRINTING AP PER CLASS- SINCE THE METRICS ARE NOT HARD CODED ANYMORE (as done in
            #  calc_batch_prediction_accuracy_per_class in metric_utils.py), THIS IS ONLY RELEVANT WHEN CHOOSING
            #  DETECTIONMETRICS, WHICH ALREADY RETURN THE METRICS VALUEST HEMSELVES AND NOT THE ITEMS REQUIRED FOR SUCH
            #  COMPUTATION. ALSO REMOVE THE BELOW LINES BY IMPLEMENTING CRITERION AS A TORCHMETRIC.

            if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
                logging_values = reduce_results_tuple_for_ddp(logging_values, next(self.net.parameters()).device)

            pbar_message_dict = get_train_loop_description_dict(logging_values, metrics, self.loss_logging_items_names)

            self.valid_monitored_values = sg_trainer_utils.update_monitored_values_dict(
                monitored_values_dict=self.valid_monitored_values, new_values_dict=pbar_message_dict
            )

            if not silent_mode and evaluation_type == EvaluationType.VALIDATION:
                progress_bar_data_loader.write("===========================================================")
                sg_trainer_utils.display_epoch_summary(
                    epoch=context.epoch, n_digits=4, train_monitored_values=self.train_monitored_values, valid_monitored_values=self.valid_monitored_values
                )
                progress_bar_data_loader.write("===========================================================")

        return logging_values

    def _instantiate_net(
        self, architecture: Union[torch.nn.Module, SgModule.__class__, str], arch_params: dict, checkpoint_params: dict, *args, **kwargs
    ) -> tuple:
        """
        Instantiates nn.Module according to architecture and arch_params, and handles pretrained weights and the required
            module manipulation (i.e head replacement).

        :param architecture: String, torch.nn.Module or uninstantiated SgModule class describing the netowrks architecture.
        :param arch_params: Architecture's parameters passed to networks c'tor.
        :param checkpoint_params: checkpoint loading related parameters dictionary with 'pretrained_weights' key,
            s.t it's value is a string describing the dataset of the pretrained weights (for example "imagenent").

        :return: instantiated netowrk i.e torch.nn.Module, architecture_class (will be none when architecture is not str)

        """
        pretrained_weights = core_utils.get_param(checkpoint_params, "pretrained_weights", default_val=None)

        if pretrained_weights is not None:
            num_classes_new_head = arch_params.num_classes
            arch_params.num_classes = PRETRAINED_NUM_CLASSES[pretrained_weights]

        if isinstance(architecture, str):
            architecture_cls = ARCHITECTURES[architecture]
            net = architecture_cls(arch_params=arch_params)
        elif isinstance(architecture, SgModule.__class__):
            net = architecture(arch_params)
        else:
            net = architecture

        if pretrained_weights:
            load_pretrained_weights(net, architecture, pretrained_weights)
            if num_classes_new_head != arch_params.num_classes:
                net.replace_head(new_num_classes=num_classes_new_head)
                arch_params.num_classes = num_classes_new_head

        return net

    def _instantiate_ema_model(self, ema_params: Mapping[str, Any]) -> ModelEMA:
        """Instantiate ema model for standard SgModule.
        :param decay_type: (str) The decay climb schedule. See EMA_DECAY_FUNCTIONS for more details.
        :param decay: The maximum decay value. As the training process advances, the decay will climb towards this value
                      according to decay_type schedule. See EMA_DECAY_FUNCTIONS for more details.
        :param kwargs: Additional parameters for the decay function. See EMA_DECAY_FUNCTIONS for more details.
        """
        logger.info(f"Using EMA with params {ema_params}")
        return ModelEMA.from_params(self.net, **ema_params)

    @property
    def get_net(self):
        """
        Getter for network.
        :return: torch.nn.Module, self.net
        """
        return self.net

    def set_net(self, net: torch.nn.Module):
        """
        Setter for network.

        :param net: torch.nn.Module, value to set net
        :return:
        """
        self.net = net

    def set_ckpt_best_name(self, ckpt_best_name):
        """
        Setter for best checkpoint filename.

        :param ckpt_best_name: str, value to set ckpt_best_name
        """
        self.ckpt_best_name = ckpt_best_name

    def set_ema(self, val: bool):
        """
        Setter for self.ema

        :param val: bool, value to set ema
        """
        self.ema = val

    def _get_context_methods(self, phase: Phase) -> ContextSgMethods:
        """
        Returns ContextSgMethods holding the methods that should be accessible through phase callbacks to the user at
         the specific phase

        :param phase: Phase, controls what methods should be returned.
        :return: ContextSgMethods holding methods from self.
        """
        if phase in [
            Phase.PRE_TRAINING,
            Phase.TRAIN_EPOCH_START,
            Phase.TRAIN_EPOCH_END,
            Phase.VALIDATION_EPOCH_END,
            Phase.VALIDATION_EPOCH_END,
            Phase.POST_TRAINING,
            Phase.VALIDATION_END_BEST_EPOCH,
        ]:
            context_methods = ContextSgMethods(
                get_net=self.get_net,
                set_net=self.set_net,
                set_ckpt_best_name=self.set_ckpt_best_name,
                reset_best_metric=self._reset_best_metric,
                validate_epoch=self._validate_epoch,
                set_ema=self.set_ema,
            )
        else:
            context_methods = ContextSgMethods()

        return context_methods

    def _init_loss_logging_names(self, loss_logging_items):
        criterion_name = self.criterion.__class__.__name__
        component_names = None
        if hasattr(self.criterion, "component_names"):
            component_names = self.criterion.component_names
        elif len(loss_logging_items) > 1:
            component_names = ["loss_" + str(i) for i in range(len(loss_logging_items))]

        if component_names is not None:
            self.loss_logging_items_names = [criterion_name + "/" + component_name for component_name in component_names]
            if self.metric_to_watch in component_names:
                self.metric_to_watch = criterion_name + "/" + self.metric_to_watch
        else:
            self.loss_logging_items_names = [criterion_name]

    @classmethod
    def quantize_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:
        """
        Perform quantization aware training (QAT) according to a recipe configuration.

        This method will instantiate all the objects specified in the recipe, build and quantize the model,
        and calibrate the quantized model. The resulting quantized model and the output of the trainer.train()
        method will be returned.

        The quantized model will be exported to ONNX along with other checkpoints.

        The call to self.quantize (see docs in the next method) is done with the created
         train_loader and valid_loader. If no calibration data loader is passed through cfg.calib_loader,
         a train data laoder with the validation transforms is used for calibration.

        :param cfg: The parsed DictConfig object from yaml recipe files or a dictionary.
        :return: A tuple containing the quantized model and the output of trainer.train() method.

        :raises ValueError: If the recipe does not have the required key `quantization_params` or
        `checkpoint_params.checkpoint_path` in it.
        :raises NotImplementedError: If the recipe requests multiple GPUs or num_gpus is not equal to 1.
        :raises ImportError: If pytorch-quantization import was unsuccessful

        """
        if _imported_pytorch_quantization_failure is not None:
            raise _imported_pytorch_quantization_failure

        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        # TRIGGER CFG MODIFYING CALLBACKS
        cfg = cls._trigger_cfg_modifying_callbacks(cfg)

        quantization_params = get_param(cfg, "quantization_params")

        if quantization_params is None:
            logger.warning("Your recipe does not include quantization_params. Using default quantization params.")
            quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params
            cfg.quantization_params = quantization_params

        if get_param(cfg.checkpoint_params, "checkpoint_path") is None and get_param(cfg.checkpoint_params, "pretrained_weights") is None:
            raise ValueError("Starting checkpoint / pretrained weights are a must for QAT finetuning.")

        num_gpus = core_utils.get_param(cfg, "num_gpus")
        multi_gpu = core_utils.get_param(cfg, "multi_gpu")
        device = core_utils.get_param(cfg, "device")
        if num_gpus != 1:
            raise NotImplementedError(
                f"Recipe requests multi_gpu={cfg.multi_gpu} and num_gpus={cfg.num_gpus}. QAT is proven to work correctly only with multi_gpu=OFF and num_gpus=1"
            )

        setup_device(device=device, multi_gpu=multi_gpu, num_gpus=num_gpus)

        # INSTANTIATE DATA LOADERS
        train_dataloader = dataloaders.get(
            name=get_param(cfg, "train_dataloader"),
            dataset_params=copy.deepcopy(cfg.dataset_params.train_dataset_params),
            dataloader_params=copy.deepcopy(cfg.dataset_params.train_dataloader_params),
        )

        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=copy.deepcopy(cfg.dataset_params.val_dataset_params),
            dataloader_params=copy.deepcopy(cfg.dataset_params.val_dataloader_params),
        )

        if "calib_dataloader" in cfg:
            calib_dataloader_name = get_param(cfg, "calib_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.calib_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.calib_dataset_params)
        else:
            calib_dataloader_name = get_param(cfg, "train_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.train_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.train_dataset_params)

            # if we use whole dataloader for calibration, don't shuffle it
            # HistogramCalibrator collection routine is sensitive to order of batches and produces slightly different results
            # if we use several batches, we don't want it to be from one class if it's sequential in dataloader
            # model is in eval mode, so BNs will not be affected
            calib_dataloader_params.shuffle = cfg.quantization_params.calib_params.num_calib_batches is not None
            # we don't need training transforms during calibration, distribution of activations will be skewed
            calib_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms

        calib_dataloader = dataloaders.get(
            name=calib_dataloader_name,
            dataset_params=calib_dataset_params,
            dataloader_params=calib_dataloader_params,
        )

        # BUILD MODEL
        model = models.get(
            model_name=cfg.arch_params.get("model_name", None) or cfg.architecture,
            num_classes=cfg.get("num_classes", None) or cfg.arch_params.num_classes,
            arch_params=cfg.arch_params,
            strict_load=cfg.checkpoint_params.strict_load,
            pretrained_weights=cfg.checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.checkpoint_params.checkpoint_path,
            load_backbone=False,
        )

        recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}
        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=get_param(cfg, "ckpt_root_dir"))

        if quantization_params.ptq_only:
            res = trainer.ptq(
                calib_loader=calib_dataloader,
                model=model,
                quantization_params=quantization_params,
                valid_loader=val_dataloader,
                valid_metrics_list=cfg.training_hyperparams.valid_metrics_list,
            )
        else:
            res = trainer.qat(
                model=model,
                quantization_params=quantization_params,
                calib_loader=calib_dataloader,
                valid_loader=val_dataloader,
                train_loader=train_dataloader,
                training_params=cfg.training_hyperparams,
                additional_qat_configs_to_log=recipe_logged_cfg,
            )

        return model, res

    def qat(
        self,
        calib_loader: DataLoader,
        model: torch.nn.Module,
        valid_loader: DataLoader,
        train_loader: DataLoader,
        training_params: Mapping = None,
        quantization_params: Mapping = None,
        additional_qat_configs_to_log: Dict = None,
        valid_metrics_list: List[Metric] = None,
    ):
        """
        Performs post-training quantization (PTQ), and then quantization-aware training (QAT).
        Exports the ONNX models (ckpt_best.pth of QAT and the calibrated model) to the checkpoints directory.

        :param calib_loader: DataLoader, data loader for calibration.

        :param model: torch.nn.Module, Model to perform QAT/PTQ on. When None, will try to use the network from
        previous self.train call(that is, if there was one - will try to use self.ema_model.ema if EMA was used,
        otherwise self.net)


        :param valid_loader: DataLoader, data loader for validation. Used both for validating the calibrated model after PTQ and during QAT.
            When None, will try to use self.valid_loader if it was set in previous self.train(..) call (default=None).

        :param train_loader: DataLoader, data loader for QA training, can be ignored when quantization_params["ptq_only"]=True (default=None).

        :param quantization_params: Mapping, with the following entries:defaults-
            selective_quantizer_params:
              calibrator_w: "max"        # calibrator type for weights, acceptable types are ["max", "histogram"]
              calibrator_i: "histogram"  # calibrator type for inputs acceptable types are ["max", "histogram"]
              per_channel: True          # per-channel quantization of weights, activations stay per-tensor by default
              learn_amax: False          # enable learnable amax in all TensorQuantizers using straight-through estimator
              skip_modules:              # optional list of module names (strings) to skip from quantization

            calib_params:
              histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators,
                                                                # acceptable types are ["percentile", "entropy", mse"],
                                                                # "max" calibrators always use "max"

              percentile: 99.99                     # percentile for all histogram calibrators with method "percentile",
                                                    # other calibrators are not affected

              num_calib_batches:                    # number of batches to use for calibration, if None, 512 / batch_size will be used
              verbose: False                        # if calibrator should be verbose


              When None, the above default config is used (default=None)


        :param training_params: Mapping, training hyper parameters for QAT, same as in super.train(...). When None, will try to use self.training_params
         which is set in previous self.train(..) call (default=None).

        :param additional_qat_configs_to_log: Dict, Optional dictionary containing configs that will be added to the QA training's
         sg_logger. Format should be {"Config_title_1": {...}, "Config_title_2":{..}}.

        :param valid_metrics_list:  (list(torchmetrics.Metric)) metrics list for evaluation of the calibrated model.
        When None, the validation metrics from training_params are used). (default=None).

        :return: Validation results of the QAT model in case quantization_params['ptq_only']=False and of the PTQ
        model otherwise.
        """

        if quantization_params is None:
            quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params
            logger.info(f"Using default quantization params: {quantization_params}")
        valid_metrics_list = valid_metrics_list or get_param(training_params, "valid_metrics_list")

        _ = self.ptq(
            calib_loader=calib_loader,
            model=model,
            quantization_params=quantization_params,
            valid_loader=valid_loader,
            valid_metrics_list=valid_metrics_list,
            deepcopy_model_for_export=True,
        )
        # TRAIN
        model.train()
        torch.cuda.empty_cache()

        res = self.train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            training_params=training_params,
            additional_configs_to_log=additional_qat_configs_to_log,
        )

        # EXPORT QUANTIZED MODEL TO ONNX
        input_shape = next(iter(valid_loader))[0].shape
        os.makedirs(self.checkpoints_dir_path, exist_ok=True)
        qdq_onnx_path = os.path.join(self.checkpoints_dir_path, f"{self.experiment_name}_{'x'.join((str(x) for x in input_shape))}_qat.onnx")

        # TODO: modify SG's convert_to_onnx for quantized models and use it instead
        export_quantized_module_to_onnx(
            model=model.cpu(),
            onnx_filename=qdq_onnx_path,
            input_shape=input_shape,
            input_size=input_shape,
            train=False,
        )
        logger.info(f"Exported QAT ONNX to {qdq_onnx_path}")
        return res

    def ptq(
        self,
        calib_loader: DataLoader,
        model: nn.Module,
        valid_loader: DataLoader,
        valid_metrics_list: List[torchmetrics.Metric],
        quantization_params: Dict = None,
        deepcopy_model_for_export: bool = False,
    ):
        """
        Performs post-training quantization (calibration of the model)..

        :param calib_loader: DataLoader, data loader for calibration.

        :param model: torch.nn.Module, Model to perform calibration on. When None, will try to use self.net which is
        set in previous self.train(..) call (default=None).

        :param valid_loader: DataLoader, data loader for validation. Used both for validating the calibrated model.
            When None, will try to use self.valid_loader if it was set in previous self.train(..) call (default=None).

        :param quantization_params: Mapping, with the following entries:defaults-
            selective_quantizer_params:
              calibrator_w: "max"        # calibrator type for weights, acceptable types are ["max", "histogram"]
              calibrator_i: "histogram"  # calibrator type for inputs acceptable types are ["max", "histogram"]
              per_channel: True          # per-channel quantization of weights, activations stay per-tensor by default
              learn_amax: False          # enable learnable amax in all TensorQuantizers using straight-through estimator
              skip_modules:              # optional list of module names (strings) to skip from quantization

            calib_params:
              histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators,
                                                                # acceptable types are ["percentile", "entropy", mse"],
                                                                # "max" calibrators always use "max"

              percentile: 99.99                     # percentile for all histogram calibrators with method "percentile",
                                                    # other calibrators are not affected

              num_calib_batches:                    # number of batches to use for calibration, if None, 512 / batch_size will be used
              verbose: False                        # if calibrator should be verbose


              When None, the above default config is used (default=None)



        :param valid_metrics_list:  (list(torchmetrics.Metric)) metrics list for evaluation of the calibrated model.

        :param deepcopy_model_for_export: bool, Whether to export deepcopy(model). Necessary in case further training is
            performed and prep_model_for_conversion makes the network un-trainable (i.e RepVGG blocks).

        :return: Validation results of the calibrated model.
        """

        if quantization_params is None:
            quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params
            logger.info(f"Using default quantization params: {quantization_params}")

        selective_quantizer_params = get_param(quantization_params, "selective_quantizer_params")
        calib_params = get_param(quantization_params, "calib_params")
        model.to(device_config.device)
        # QUANTIZE MODEL
        model.eval()
        fuse_repvgg_blocks_residual_branches(model)
        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights=get_param(selective_quantizer_params, "calibrator_w"),
            default_quant_modules_calibrator_inputs=get_param(selective_quantizer_params, "calibrator_i"),
            default_per_channel_quant_weights=get_param(selective_quantizer_params, "per_channel"),
            default_learn_amax=get_param(selective_quantizer_params, "learn_amax"),
            verbose=get_param(calib_params, "verbose"),
        )
        q_util.register_skip_quantization(layer_names=get_param(selective_quantizer_params, "skip_modules"))
        q_util.quantize_module(model)
        # CALIBRATE MODEL
        logger.info("Calibrating model...")
        calibrator = QuantizationCalibrator(
            verbose=get_param(calib_params, "verbose"),
            torch_hist=True,
        )
        calibrator.calibrate_model(
            model,
            method=get_param(calib_params, "histogram_calib_method"),
            calib_data_loader=calib_loader,
            num_calib_batches=get_param(calib_params, "num_calib_batches") or len(calib_loader),
            percentile=get_param(calib_params, "percentile", 99.99),
        )
        calibrator.reset_calibrators(model)  # release memory taken by calibrators
        # VALIDATE PTQ MODEL AND PRINT SUMMARY
        logger.info("Validating PTQ model...")
        valid_metrics_dict = self.test(model=model, test_loader=valid_loader, test_metrics_list=valid_metrics_list)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in valid_metrics_dict.items()]
        logger.info("\n".join(results))

        input_shape = next(iter(valid_loader))[0].shape
        os.makedirs(self.checkpoints_dir_path, exist_ok=True)
        qdq_onnx_path = os.path.join(self.checkpoints_dir_path, f"{self.experiment_name}_{'x'.join((str(x) for x in input_shape))}_ptq.onnx")

        # TODO: modify SG's convert_to_onnx for quantized models and use it instead
        export_quantized_module_to_onnx(
            model=model.cpu(),
            onnx_filename=qdq_onnx_path,
            input_shape=input_shape,
            input_size=input_shape,
            train=False,
            deepcopy_model=deepcopy_model_for_export,
        )

        return valid_metrics_dict
