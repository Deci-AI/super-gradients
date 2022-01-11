import os
import sys
from copy import deepcopy
from enum import Enum
from typing import Union, Tuple

import numpy as np
import pkg_resources
import torch
import torchvision.transforms as transforms
from deprecated import deprecated
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import MetricCollection
from tqdm import tqdm
from piptools.scripts.sync import _get_installed_distributions

from super_gradients.common.environment import env_helpers
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers import SG_LOGGERS
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.training import ARCHITECTURES, utils as core_utils
from super_gradients.training.utils import sg_model_utils
from super_gradients.training import metrics
from super_gradients.training.exceptions.sg_model_exceptions import UnsupportedOptimizerFormat
from super_gradients.training.datasets import DatasetInterface
from super_gradients.training.losses import LOSSES
from super_gradients.training.metrics.metric_utils import get_metrics_titles, get_metrics_results_tuple, \
    get_logging_values, \
    get_metrics_dict, get_train_loop_description_dict
from super_gradients.training.models import SgModule
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.utils.distributed_training_utils import MultiGPUModeAutocastWrapper, \
    reduce_results_tuple_for_ddp, compute_precise_bn_stats
from super_gradients.training.utils.ema import ModelEMA
from super_gradients.training.utils.optimizer_utils import build_optimizer
from super_gradients.training.utils.weight_averaging_utils import ModelWeightAveraging
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils import random_seed
from super_gradients.training.utils.checkpoint_utils import get_ckpt_local_path, read_ckpt_state_dict, \
    load_checkpoint_to_model, load_pretrained_weights
from super_gradients.training.datasets.datasets_utils import DatasetStatisticsTensorboardLogger
from super_gradients.training.utils.callbacks import CallbackHandler, Phase, LR_SCHEDULERS_CLS_DICT, PhaseContext, \
    MetricsUpdateCallback, LR_WARMUP_CLS_DICT
from super_gradients.common.environment import environment_config
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES

logger = get_logger(__name__)


class StrictLoad(Enum):
    """
    Wrapper for adding more functionality to torch's strict_load parameter in load_state_dict().
        Attributes:
            OFF              - Native torch "strict_load = off" behaviour. See nn.Module.load_state_dict() documentation for more details.
            ON               - Native torch "strict_load = on" behaviour. See nn.Module.load_state_dict() documentation for more details.
            NO_KEY_MATCHING  - Allows the usage of SuperGradient's adapt_checkpoint function, which loads a checkpoint by matching each
                               layer's shapes (and bypasses the strict matching of the names of each layer (ie: disregards the state_dict key matching)).
    """
    OFF = False
    ON = True
    NO_KEY_MATCHING = 'no_key_matching'


class MultiGPUMode(str, Enum):
    """
    MultiGPUMode

        Attributes:
            OFF                       - Single GPU Mode / CPU Mode
            DATA_PARALLEL             - Multiple GPUs, Synchronous
            DISTRIBUTED_DATA_PARALLEL - Multiple GPUs, Asynchronous
    """
    OFF = 'Off'
    DATA_PARALLEL = 'DP'
    DISTRIBUTED_DATA_PARALLEL = 'DDP'
    AUTO = "AUTO"


class EvaluationType(str, Enum):
    """
    EvaluationType

    Passed to SgModel.evaluate(..), and controls which phase callbacks should be triggered (if at all).

        Attributes:
            TEST
            VALIDATION

    """
    TEST = 'TEST'
    VALIDATION = 'VALIDATION'


class SgModel:
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

    def __init__(self, experiment_name: str, device: str = None, multi_gpu: Union[MultiGPUMode, str] = MultiGPUMode.AUTO,
                 model_checkpoints_location: str = 'local',
                 overwrite_local_checkpoint: bool = True, ckpt_name: str = 'ckpt_latest.pth',
                 post_prediction_callback: DetectionPostPredictionCallback = None, ckpt_root_dir=None):
        """

        :param experiment_name:                      Used for logging and loading purposes
        :param device:                          If equal to 'cpu' runs on the CPU otherwise on GPU
        :param multi_gpu:                       If True, runs on all available devices
        :param model_checkpoints_location:      If set to 's3' saves the Checkpoints in AWS S3
                                                otherwise saves the Checkpoints Locally
        :param overwrite_local_checkpoint:      If set to False keeps the current local checkpoint when importing
                                                checkpoint from cloud service, otherwise overwrites the local checkpoints file
        :param ckpt_name:                       The Checkpoint to Load
        :ckpt_root_dir:                         Local root directory path where all experiment logging directories will
                                                 reside. When none is give, it is assumed that
                                                 pkg_resources.resource_filename('checkpoints', "") exists and will be used.

        """
        # SET THE EMPTY PROPERTIES
        self.net, self.architecture, self.arch_params, self.dataset_interface = None, None, None, None
        self.architecture_cls, self.device, self.multi_gpu = None, None, None
        self.dataset_params, self.train_loader, self.valid_loader, self.test_loader, self.classes = None, None, None, None, None
        self.ema = None
        self.ema_model = None
        self.sg_logger = None
        self.update_param_groups = None
        self.post_prediction_callback = None
        self.criterion = None
        self.training_params = None
        self.scaler = None
        self.phase_callbacks = None

        # SET THE DEFAULT PROPERTIES
        self.half_precision = False
        self.load_checkpoint = False
        self.load_backbone = False
        self.load_weights_only = False
        self.ddp_silent_mode = False
        self.source_ckpt_folder_name = None
        self.model_weight_averaging = None
        self.average_model_checkpoint_filename = 'average_model.pth'
        self.start_epoch = 0
        self.best_metric = np.inf
        self.external_checkpoint_path = None

        # DETERMINE THE LOCATION OF THE LOSS AND ACCURACY IN THE RESULTS TUPLE OUTPUTED BY THE TEST
        self.loss_idx_in_results_tuple, self.acc_idx_in_results_tuple = None, None

        # METRICS
        self.loss_logging_items_names = None
        self.train_metrics = None
        self.valid_metrics = None
        self.greater_metric_to_watch_is_better = None

        # SETTING THE PROPERTIES FROM THE CONSTRUCTOR
        self.experiment_name = experiment_name
        self.ckpt_name = ckpt_name
        self.overwrite_local_checkpoint = overwrite_local_checkpoint
        self.model_checkpoints_location = model_checkpoints_location

        # CREATING THE LOGGING DIR BASED ON THE INPUT PARAMS TO PREVENT OVERWRITE OF LOCAL VERSION
        if ckpt_root_dir:
            self.checkpoints_dir_path = os.path.join(ckpt_root_dir, self.experiment_name)
        elif pkg_resources.resource_exists("checkpoints", ""):
            self.checkpoints_dir_path = pkg_resources.resource_filename('checkpoints', self.experiment_name)
        else:
            raise ValueError("Illegal checkpoints directory: pass ckpt_root_dir that exists, or add 'checkpoints' to"
                             "resources.")

        # INITIALIZE THE DEVICE FOR THE MODEL
        self._initialize_device(requested_device=device, requested_multi_gpu=multi_gpu)

        self.post_prediction_callback = post_prediction_callback
        # SET THE DEFAULTS
        # TODO: SET DEFAULT TRAINING PARAMS FOR EACH TASK

        default_results_titles = ['Train Loss', 'Train Acc', 'Train Top5', 'Valid Loss', 'Valid Acc', 'Valid Top5']

        self.results_titles = default_results_titles

        self.loss_idx_in_results_tuple, self.acc_idx_in_results_tuple = 0, 1
        default_train_metrics, default_valid_metrics = MetricCollection([Accuracy(), Top5()]), MetricCollection(
            [Accuracy(), Top5()])

        default_loss_logging_items_names = ["Loss"]

        self.train_metrics, self.valid_metrics = default_train_metrics, default_valid_metrics
        self.loss_logging_items_names = default_loss_logging_items_names

    def connect_dataset_interface(self, dataset_interface: DatasetInterface, data_loader_num_workers: int = 8):
        """
        :param dataset_interface: DatasetInterface object
        :param data_loader_num_workers: The number of threads to initialize the Data Loaders with
            The dataset to be connected
        """
        self.dataset_interface = dataset_interface
        self.train_loader, self.valid_loader, self.test_loader, self.classes = \
            self.dataset_interface.get_data_loaders(batch_size_factor=self.num_devices,
                                                    num_workers=data_loader_num_workers,
                                                    distributed_sampler=self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL)

        self.dataset_params = self.dataset_interface.get_dataset_params()

    # FIXME - we need to resolve flake8's 'function is too complex' for this function
    def build_model(self,  # noqa: C901 - too complex
                    architecture: Union[str, nn.Module],
                    arch_params={},
                    load_checkpoint: bool = False,
                    strict_load: StrictLoad = StrictLoad.ON,
                    source_ckpt_folder_name: str = None,
                    load_weights_only: bool = False,
                    load_backbone: bool = False,
                    external_checkpoint_path: str = None,
                    load_ema_as_net: bool = False):
        """
        :param architecture:               Defines the network's architecture from models/ALL_ARCHITECTURES
        :param arch_params:                Architecture H.P. e.g.: block, num_blocks, num_classes, etc.
        :param load_checkpoint:            Load a pre-trained checkpoint
        :param strict_load:                See StrictLoad class documentation for details.
        :param source_ckpt_folder_name:    folder name to load the checkpoint from (self.experiment_name if none is given)
        :param load_weights_only:          loads only the weight from the checkpoint and zeroize the training params
        :param load_backbone:              loads the provided checkpoint to self.net.backbone instead of self.net
        :param external_checkpoint_path:   The path to the external checkpoint to be loaded. Can be absolute or relative
                                           (ie: path/to/checkpoint.pth). If provided, will automatically attempt to
                                           load the checkpoint even if the load_checkpoint flag is not provided.
        """
        if 'num_classes' not in arch_params.keys():
            if self.dataset_interface is None:
                raise Exception('Error', 'Number of classes not defined in arch params and dataset is not defined')
            else:
                arch_params['num_classes'] = len(self.classes)

        self.arch_params = core_utils.HpmStruct(**arch_params)

        pretrained_weights = core_utils.get_param(self.arch_params, 'pretrained_weights', default_val=None)
        if pretrained_weights is not None:
            num_classes_new_head = self.arch_params.num_classes
            self.arch_params.num_classes = PRETRAINED_NUM_CLASSES[pretrained_weights]

        # OVERRIDE THE INPUT PARAMS WITH THE arch_params VALUES
        load_weights_only = core_utils.get_param(self.arch_params, 'load_weights_only', default_val=load_weights_only)

        self.source_ckpt_folder_name = core_utils.get_param(self.arch_params, 'source_ckpt_folder_name',
                                                            default_val=source_ckpt_folder_name)
        strict_load = core_utils.get_param(self.arch_params, 'strict_load', default_val=strict_load)

        self.arch_params.sync_bn = core_utils.get_param(self.arch_params, 'sync_bn', default_val=False)
        self.load_checkpoint = load_checkpoint or core_utils.get_param(self.arch_params, 'load_checkpoint',
                                                                       default_val=False)
        self.load_backbone = core_utils.get_param(self.arch_params, 'load_backbone', default_val=load_backbone)
        self.external_checkpoint_path = core_utils.get_param(self.arch_params, 'external_checkpoint_path',
                                                             default_val=external_checkpoint_path)

        if isinstance(architecture, str):
            self.architecture_cls = ARCHITECTURES[architecture]
            self.net = self.architecture_cls(arch_params=self.arch_params)
        elif isinstance(architecture, SgModule.__class__):
            self.net = architecture(self.arch_params)
        else:
            self.net = architecture

        # SAVE THE ARCHITECTURE FOR NEURAL ARCHITECTURE SEARCH
        if hasattr(self.net, 'structure'):
            self.architecture = self.net.structure

        self.net.to(self.device)

        # FOR MULTI-GPU TRAINING (not distributed)
        if self.multi_gpu == MultiGPUMode.DATA_PARALLEL:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.device_ids)
        elif self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            if self.arch_params.sync_bn:
                if not self.ddp_silent_mode:
                    logger.info('DDP - Using Sync Batch Norm... Training time will be affected accordingly')
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net).to(self.device)

            local_rank = int(self.device.split(':')[1])
            self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                                 device_ids=[local_rank],
                                                                 output_device=local_rank,
                                                                 find_unused_parameters=True)

        else:
            self.net = core_utils.WrappedModel(self.net)

        # SET THE FLAG FOR DIFFERENT PARAMETER GROUP OPTIMIZER UPDATE
        self.update_param_groups = hasattr(self.net.module, 'update_param_groups')

        # LOAD AN EXISTING CHECKPOINT IF INDICATED
        self.checkpoint = {}

        if self.load_checkpoint or self.external_checkpoint_path:
            self.load_weights_only = load_weights_only
            self._load_checkpoint_to_model(strict=strict_load, load_backbone=self.load_backbone,
                                           source_ckpt_folder_name=self.source_ckpt_folder_name,
                                           load_ema_as_net=load_ema_as_net)
        if pretrained_weights:
            load_pretrained_weights(self.net, architecture, pretrained_weights)
            if num_classes_new_head != self.arch_params.num_classes:
                self.net.module.replace_head(new_num_classes=num_classes_new_head)
                self.arch_params.num_classes = num_classes_new_head
                self.net.to(self.device)

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
        progress_bar_train_loader = tqdm(self.train_loader, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True,
                                         disable=silent_mode)
        progress_bar_train_loader.set_description(f"Train epoch {epoch}")

        # RESET/INIT THE METRIC LOGGERS
        self.train_metrics.reset()
        self.train_metrics.to(self.device)
        loss_avg_meter = core_utils.utils.AverageMeter()

        context = PhaseContext(epoch=epoch,
                               optimizer=self.optimizer,
                               metrics_compute_fn=self.train_metrics,
                               loss_avg_meter=loss_avg_meter,
                               criterion=self.criterion,
                               device=self.device,
                               lr_warmup_epochs=self.training_params.lr_warmup_epochs)

        for batch_idx, batch_items in enumerate(progress_bar_train_loader):
            batch_items = core_utils.tensor_container_to_device(batch_items, self.device, non_blocking=True)
            inputs, targets, additional_batch_items = sg_model_utils.unpack_batch_items(batch_items)
            # AUTOCAST IS ENABLED ONLY IF self.training_params.mixed_precision - IF enabled=False AUTOCAST HAS NO EFFECT
            with autocast(enabled=self.training_params.mixed_precision):
                # FORWARD PASS TO GET NETWORK'S PREDICTIONS
                outputs = self.net(inputs)

                # COMPUTE THE LOSS FOR BACK PROP + EXTRA METRICS COMPUTED DURING THE LOSS FORWARD PASS
                loss, loss_log_items = self._get_losses(outputs, targets)

            context.update_context(batch_idx=batch_idx,
                                   inputs=inputs,
                                   preds=outputs,
                                   target=targets,
                                   loss_log_items=loss_log_items,
                                   **additional_batch_items)

            self.phase_callback_handler(Phase.TRAIN_BATCH_END, context)

            # LOG LR THAT WILL BE USED IN CURRENT EPOCH AND AFTER FIRST WARMUP/LR_SCHEDULER UPDATE BEFORE WEIGHT UPDATE
            if not self.ddp_silent_mode and batch_idx == 0:
                self._write_lrs(epoch)

            self.backward_step(loss, epoch, batch_idx, context)

            # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
            logging_values = loss_avg_meter.average + get_metrics_results_tuple(self.train_metrics)
            gpu_memory_utilization = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0

            # RENDER METRICS PROGRESS
            pbar_message_dict = get_train_loop_description_dict(logging_values,
                                                                self.train_metrics,
                                                                self.loss_logging_items_names,
                                                                gpu_mem=gpu_memory_utilization)

            progress_bar_train_loader.set_postfix(**pbar_message_dict)

        if not self.ddp_silent_mode:
            self.sg_logger.upload()

        return logging_values

    def _get_losses(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        # GET THE OUTPUT OF THE LOSS FUNCTION
        loss = self.criterion(outputs, targets)
        if isinstance(loss, tuple):
            loss, loss_logging_items = loss
            # IF ITS NOT A TUPLE THE LOGGING ITEMS CONTAIN ONLY THE LOSS FOR BACKPROP (USER DEFINED LOSS RETURNS SCALAR)
        else:
            loss_logging_items = loss.unsqueeze(0).detach()

        if len(loss_logging_items) != len(self.loss_logging_items_names):
            raise ValueError("Loss output length must match loss_logging_items_names. Got " + str(
                len(loss_logging_items)) + ', and ' + str(len(self.loss_logging_items_names)))
        # RETURN AND THE LOSS LOGGING ITEMS COMPUTED DURING LOSS FORWARD PASS
        return loss, loss_logging_items

    def backward_step(self, loss: torch.Tensor, epoch: int, batch_idx: int, context: PhaseContext):
        """
        Run backprop on the loss and perform a step
        :param loss: The value computed by the loss function
        :param optimizer: An object that can perform a gradient step and zeroize model gradient
        :param epoch: number of epoch the training is on
        :param batch_idx: number of iteration inside the current epoch
        :param context: current phase context
        :return:
        """
        # SCALER IS ENABLED ONLY IF self.training_params.mixed_precision=True
        self.scaler.scale(loss).backward()

        # ACCUMULATE GRADIENT FOR X BATCHES BEFORE OPTIMIZING
        integrated_batches_num = batch_idx + len(self.train_loader) * epoch + 1

        if integrated_batches_num % self.batch_accumulate == 0:
            # SCALER IS ENABLED ONLY IF self.training_params.mixed_precision=True
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()
            if self.ema:
                self.ema_model.update(self.net, integrated_batches_num / (len(self.train_loader) * self.max_epochs))

            # RUN PHASE CALLBACKS
            self.phase_callback_handler(Phase.TRAIN_BATCH_STEP, context)

    def save_checkpoint(self, optimizer=None, epoch: int = None, validation_results_tuple: tuple = None, context: PhaseContext = None):
        """
        Save the current state dict as latest (always), best (if metric was improved), epoch# (if determined in training
        params)
        """
        # WHEN THE validation_results_tuple IS NONE WE SIMPLY SAVE THE state_dict AS LATEST AND Return
        if validation_results_tuple is None:
            self.sg_logger.add_checkpoint(tag='ckpt_latest_weights_only.pth', state_dict={'net': self.net.state_dict()},
                                          global_step=epoch)
            return

        # COMPUTE THE CURRENT metric
        # IF idx IS A LIST - SUM ALL THE VALUES STORED IN THE LIST'S INDICES
        metric = validation_results_tuple[self.metric_idx_in_results_tuple] if isinstance(
            self.metric_idx_in_results_tuple, int) else \
            sum([validation_results_tuple[idx] for idx in self.metric_idx_in_results_tuple])

        # BUILD THE state_dict
        state = {'net': self.net.state_dict(), 'acc': metric, 'epoch': epoch}
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()

        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()

        if self.ema:
            state['ema_net'] = self.ema_model.ema.state_dict()
        # SAVES CURRENT MODEL AS ckpt_latest
        self.sg_logger.add_checkpoint(tag='ckpt_latest.pth', state_dict=state, global_step=epoch)

        # SAVE MODEL AT SPECIFIC EPOCHS DETERMINED BY save_ckpt_epoch_list
        if epoch in self.training_params.save_ckpt_epoch_list:
            self.sg_logger.add_checkpoint(tag=f'ckpt_epoch_{epoch}.pth', state_dict=state, global_step=epoch)

        # OVERRIDE THE BEST CHECKPOINT AND best_metric IF metric GOT BETTER THAN THE PREVIOUS BEST
        if (metric > self.best_metric and self.greater_metric_to_watch_is_better) or (
                metric < self.best_metric and not self.greater_metric_to_watch_is_better):
            # STORE THE CURRENT metric AS BEST
            self.best_metric = metric
            self.sg_logger.add_checkpoint(tag='ckpt_best.pth', state_dict=state, global_step=epoch)

            # RUN PHASE CALLBACKS
            self.phase_callback_handler(Phase.VALIDATION_END_BEST_EPOCH, context)

            if isinstance(metric, torch.Tensor):
                metric = metric.item()
            logger.info("Best checkpoint overriden: validation " + self.metric_to_watch + ": " + str(metric))

        if self.training_params.average_best_models:
            net_for_averaging = self.ema_model.ema if self.ema else self.net
            averaged_model_sd = self.model_weight_averaging.get_average_model(net_for_averaging,
                                                                              validation_results_tuple=validation_results_tuple)
            self.sg_logger.add_checkpoint(tag=self.average_model_checkpoint_filename,
                                          state_dict={'net': averaged_model_sd}, global_step=epoch)

    # FIXME - we need to resolve flake8's 'function is too complex' for this function
    def train(self, training_params: dict = dict()):  # noqa: C901
        """

        train - Trains the Model

        IMPORTANT NOTE: Additional batch parameters can be added as a third item (optional) if a tuple is returned by
          the data loaders, as dictionary. The phase context will hold the additional items, under an attribute with
          the same name as the key in this dictionary. Then such items can be accessed through phase callbacks.


            :param training_params:
                - `max_epochs` : int

                    Number of epochs to run training.

                - `lr_updates` : list(int)

                    List of fixed epoch numbers to perform learning rate updates when `lr_mode='step'`.

                - `lr_decay_factor` : float

                    Decay factor to apply to the learning rate at each update when `lr_mode='step'`.


                -  `lr_mode` : str

                    Learning rate scheduling policy, one of ['step','poly','cosine','function']. 'step' refers to
                    constant updates at epoch numbers passed through `lr_updates`. 'cosine' refers to Cosine Anealing
                    policy as mentioned in https://arxiv.org/abs/1608.03983. 'poly' refers to polynomial decrease i.e
                    in each epoch iteration `self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)),
                    0.9)` 'function' refers to user defined learning rate scheduling function, that is passed through
                    `lr_schedule_function`.

                - `lr_schedule_function` : Union[callable,None]

                    Learning rate scheduling function to be used when `lr_mode` is 'function'.

                - `lr_warmup_epochs` : int (default=0)

                    Number of epochs for learning rate warm up - see https://arxiv.org/pdf/1706.02677.pdf (Section 2.2).

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
                              "yolo_v5_loss": YoLoV5DetectionLoss,
                              "ssd_loss": SSDLoss,


                    or user defined nn.module loss function.

                    IMPORTANT: forward(...) should return a (loss, loss_items) tuple where loss is the tensor used
                    for backprop (i.e what your original loss function returns), and loss_items should be a tensor of
                    shape (n_items), of values computed during the forward pass which we desire to log over the
                    entire epoch. For example- the loss itself should always be logged. Another example is a scenario
                    where the computed loss is the sum of a few components we would like to log- these entries in
                    loss_items).

                    When training, set the loss_logging_items_names parameter in train_params to be a list of
                    strings, of length n_items who's ith element is the name of the ith entry in loss_items. Then
                    each item will be logged, rendered on tensorboard and "watched" (i.e saving model checkpoints
                    according to it).

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
                        loss function's forward pass.

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

                -  `save_full_train_log` : bool (default=False)

                    When set, a full log (of all super_gradients modules, including uncaught exceptions from any other
                     module) of the training will be saved in the checkpoint directory under full_train_log.log

                -  `sg_logger` : Union[AbstractSGLogger, str] (defauls=base_sg_logger)

                    Define the SGLogger object for this training process. The SGLogger handles all disk writes, logs, TensorBoard, remote logging
                    and remote storage. By overriding the default base_sg_logger, you can change the storage location, support external monitoring and logging
                    or support remote storage.

                -   `sg_logger_params` : dict

                    SGLogger parameters
        :return:
        """
        global logger

        if self.net is None:
            raise Exception('Model', 'No model found')
        if self.dataset_interface is None:
            raise Exception('Data', 'No dataset found')

        self.training_params = TrainingParams()
        self.training_params.override(**training_params)

        # SET RANDOM SEED
        random_seed(is_ddp=self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL,
                    device=self.device, seed=self.training_params.seed)

        silent_mode = self.training_params.silent_mode or self.ddp_silent_mode
        # METRICS
        self.train_metrics = MetricCollection(self.training_params.train_metrics_list)
        self.valid_metrics = MetricCollection(self.training_params.valid_metrics_list)
        self.loss_logging_items_names = self.training_params.loss_logging_items_names

        self.results_titles = ["Train_" + t for t in
                               self.loss_logging_items_names + get_metrics_titles(self.train_metrics)] + \
                              ["Valid_" + t for t in
                               self.loss_logging_items_names + get_metrics_titles(self.valid_metrics)]

        # Store the metric to follow (loss\accuracy) and initialize as the worst value
        self.metric_to_watch = self.training_params.metric_to_watch
        self.greater_metric_to_watch_is_better = self.training_params.greater_metric_to_watch_is_better
        self.metric_idx_in_results_tuple = (self.loss_logging_items_names + get_metrics_titles(self.valid_metrics)).index(self.metric_to_watch)

        # Allowing loading instantiated loss or string
        if isinstance(self.training_params.loss, str):
            criterion_cls = LOSSES[self.training_params.loss]
            self.criterion = criterion_cls(**self.training_params.criterion_params)

        elif isinstance(self.training_params.loss, nn.Module):
            self.criterion = self.training_params.loss

        self.criterion.to(self.device)

        self.max_epochs = self.training_params.max_epochs

        self.ema = self.training_params.ema

        self.precise_bn = self.training_params.precise_bn
        self.precise_bn_batch_size = self.training_params.precise_bn_batch_size

        self.batch_accumulate = self.training_params.batch_accumulate
        num_batches = len(self.train_loader)

        if self.ema:
            ema_params = self.training_params.ema_params
            logger.info(f'Using EMA with params {ema_params}')
            self.ema_model = ModelEMA(self.net, **ema_params)
            self.ema_model.updates = self.start_epoch * num_batches // self.batch_accumulate
            if self.load_checkpoint:
                if 'ema_net' in self.checkpoint.keys():
                    self.ema_model.ema.load_state_dict(self.checkpoint['ema_net'])
                else:
                    self.ema = False
                    logger.warning(
                        "[Warning] Checkpoint does not include EMA weights, continuing training without EMA.")

        self.run_validation_freq = self.training_params.run_validation_freq
        validation_results_tuple = (0, 0)
        inf_time = 0
        timer = core_utils.Timer(self.device)

        # IF THE LR MODE IS NOT DEFAULT TAKE IT FROM THE TRAINING PARAMS
        self.lr_mode = self.training_params.lr_mode
        load_opt_params = self.training_params.load_opt_params

        self.phase_callbacks = self.training_params.phase_callbacks

        if self.lr_mode is not None:
            sg_lr_callback_cls = LR_SCHEDULERS_CLS_DICT[self.lr_mode]
            self.phase_callbacks.append(sg_lr_callback_cls(train_loader_len=len(self.train_loader),
                                                           net=self.net,
                                                           training_params=self.training_params,
                                                           update_param_groups=self.update_param_groups,
                                                           **self.training_params.to_dict()))
        if self.training_params.lr_warmup_epochs > 0:
            warmup_callback_cls = LR_WARMUP_CLS_DICT[self.training_params.warmup_mode]
            self.phase_callbacks.append(warmup_callback_cls(train_loader_len=len(self.train_loader),
                                                            net=self.net,
                                                            training_params=self.training_params,
                                                            update_param_groups=self.update_param_groups,
                                                            **self.training_params.to_dict()))

        self.phase_callbacks.append(MetricsUpdateCallback(Phase.TRAIN_BATCH_END))
        self.phase_callbacks.append(MetricsUpdateCallback(Phase.VALIDATION_BATCH_END))

        self.phase_callback_handler = CallbackHandler(callbacks=self.phase_callbacks)

        if not self.ddp_silent_mode:
            self._initialize_sg_logger_objects()

            if self.training_params.dataset_statistics:
                dataset_statistics_logger = DatasetStatisticsTensorboardLogger(self.sg_logger)
                dataset_statistics_logger.analyze(self.train_loader, dataset_params=self.dataset_params,
                                                  title="Train-set", anchors=self.net.module.arch_params.anchors)
                dataset_statistics_logger.analyze(self.valid_loader, dataset_params=self.dataset_params,
                                                  title="val-set")
            # AVERAGE BEST 10 MODELS PARAMS
            if self.training_params.average_best_models:
                self.model_weight_averaging = ModelWeightAveraging(self.checkpoints_dir_path,
                                                                   greater_is_better=self.greater_metric_to_watch_is_better,
                                                                   source_ckpt_folder_name=self.source_ckpt_folder_name,
                                                                   metric_to_watch=self.metric_to_watch,
                                                                   metric_idx=self.metric_idx_in_results_tuple,
                                                                   load_checkpoint=self.load_checkpoint,
                                                                   model_checkpoints_location=self.model_checkpoints_location)
        if self.training_params.save_full_train_log and not self.ddp_silent_mode:
            logger = get_logger(__name__,
                                training_log_path=self.sg_logger.log_file_path.replace('.txt', 'full_train_log.log'))
            sg_model_utils.log_uncaught_exceptions(logger)

        if not self.load_checkpoint or self.load_weights_only:
            # WHEN STARTING TRAINING FROM SCRATCH, DO NOT LOAD OPTIMIZER PARAMS (EVEN IF LOADING BACKBONE)
            self.start_epoch = 0
            self.best_metric = -1 * np.inf if self.greater_metric_to_watch_is_better else np.inf
            load_opt_params = False

        if isinstance(self.training_params.optimizer, str) and self.training_params.optimizer in ['Adam', 'SGD',
                                                                                                  'RMSProp',
                                                                                                  'RMSpropTF']:
            self.optimizer = build_optimizer(net=self.net, lr=self.training_params.initial_lr,
                                             training_params=self.training_params)
        elif isinstance(self.training_params.optimizer, torch.optim.Optimizer):
            self.optimizer = self.training_params.optimizer
        else:
            raise UnsupportedOptimizerFormat()

        if self.load_checkpoint and load_opt_params:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        self._initialize_mixed_precision(self.training_params.mixed_precision)

        context = PhaseContext(optimizer=self.optimizer, net=self.net, experiment_name=self.experiment_name,
                               ckpt_dir=self.checkpoints_dir_path,
                               lr_warmup_epochs=self.training_params.lr_warmup_epochs, sg_logger=self.sg_logger)
        self.phase_callback_handler(Phase.PRE_TRAINING, context)

        try:
            # HEADERS OF THE TRAINING PROGRESS
            if not silent_mode:
                logger.info(
                    f'Started training for {self.max_epochs - self.start_epoch} epochs ({self.start_epoch}/'f'{self.max_epochs - 1})\n')
            for epoch in range(self.start_epoch, self.max_epochs):
                if context.stop_training:
                    logger.info("Request to stop training has been received, stopping training")
                    break

                # Phase.TRAIN_EPOCH_START
                # RUN PHASE CALLBACKS
                context.update_context(epoch=epoch)
                self.phase_callback_handler(Phase.TRAIN_EPOCH_START, context)

                # IN DDP- SET_EPOCH WILL CAUSE EVERY PROCESS TO BE EXPOSED TO THE ENTIRE DATASET BY SHUFFLING WITH A
                # DIFFERENT SEED EACH EPOCH START
                if self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
                    self.train_loader.sampler.set_epoch(epoch)

                train_metrics_tuple = self._train_epoch(epoch=epoch, silent_mode=silent_mode)

                # Phase.TRAIN_EPOCH_END
                # RUN PHASE CALLBACKS
                train_metrics_dict = get_metrics_dict(train_metrics_tuple, self.train_metrics,
                                                      self.loss_logging_items_names)

                context.update_context(metrics_dict=train_metrics_dict)
                self.phase_callback_handler(Phase.TRAIN_EPOCH_END, context)

                # CALCULATE PRECISE BATCHNORM STATS
                if self.precise_bn:
                    compute_precise_bn_stats(model=self.net, loader=self.train_loader,
                                             precise_bn_batch_size=self.precise_bn_batch_size,
                                             num_gpus=self.num_devices)
                    if self.ema:
                        compute_precise_bn_stats(model=self.ema_model.ema, loader=self.train_loader,
                                                 precise_bn_batch_size=self.precise_bn_batch_size,
                                                 num_gpus=self.num_devices)

                # model switch - we replace self.net.module with the ema model for the testing and saving part
                # and then switch it back before the next training epoch
                if self.ema:
                    self.ema_model.update_attr(self.net)
                    keep_model = self.net
                    self.net = self.ema_model.ema

                # RUN TEST ON VALIDATION SET EVERY self.run_validation_freq EPOCHS
                if (epoch + 1) % self.run_validation_freq == 0:
                    timer.start()
                    validation_results_tuple = self._validate_epoch(epoch=epoch, silent_mode=silent_mode)
                    inf_time = timer.stop()

                    # Phase.VALIDATION_EPOCH_END
                    # RUN PHASE CALLBACKS
                    valid_metrics_dict = get_metrics_dict(validation_results_tuple, self.valid_metrics,
                                                          self.loss_logging_items_names)

                    context.update_context(metrics_dict=valid_metrics_dict)
                    self.phase_callback_handler(Phase.VALIDATION_EPOCH_END, context)

                if self.ema:
                    self.net = keep_model

                if not self.ddp_silent_mode:
                    # SAVING AND LOGGING OCCURS ONLY IN THE MAIN PROCESS (IN CASES THERE ARE SEVERAL PROCESSES - DDP)
                    self._write_to_disk_operations(train_metrics_tuple, validation_results_tuple, inf_time, epoch, context)

            # Evaluating the average model and removing snapshot averaging file if training is completed
            if self.training_params.average_best_models:
                self._validate_final_average_model(cleanup_snapshots_pkl_file=True)

        except KeyboardInterrupt:
            logger.info(
                '\n[MODEL TRAINING EXECUTION HAS BEEN INTERRUPTED]... Please wait until SOFT-TERMINATION process '
                'finishes and saves all of the Model Checkpoints and log files before terminating...')
            logger.info('For HARD Termination - Stop the process again')

        finally:
            if self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
                # CLEAN UP THE MULTI-GPU PROCESS GROUP WHEN DONE
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()

            # PHASE.TRAIN_END
            self.phase_callback_handler(Phase.POST_TRAINING, context)

            if not self.ddp_silent_mode:
                if self.model_checkpoints_location != 'local':
                    logger.info('[CLEANUP] - Saving Checkpoint files')
                    self.sg_logger.upload()

                self.sg_logger.close()

    def _initialize_mixed_precision(self, mixed_precision_enabled: bool):
        # SCALER IS ALWAYS INITIALIZED BUT IS DISABLED IF MIXED PRECISION WAS NOT SET
        self.scaler = GradScaler(enabled=mixed_precision_enabled)

        if mixed_precision_enabled:
            assert self.device.startswith('cuda'), "mixed precision is not available for CPU"
            if self.multi_gpu == MultiGPUMode.DATA_PARALLEL:
                # IN DATAPARALLEL MODE WE NEED TO WRAP THE FORWARD FUNCTION OF OUR MODEL SO IT WILL RUN WITH AUTOCAST.
                # BUT SINCE THE MODULE IS CLONED TO THE DEVICES ON EACH FORWARD CALL OF A DATAPARALLEL MODEL,
                # WE HAVE TO REGISTER THE WRAPPER BEFORE EVERY FORWARD CALL
                def hook(module, _):
                    module.forward = MultiGPUModeAutocastWrapper(module.forward)

                self.net.module.register_forward_pre_hook(hook=hook)

            if self.load_checkpoint:
                scaler_state_dict = core_utils.get_param(self.checkpoint, 'scaler_state_dict')
                if scaler_state_dict is None:
                    logger.warning(
                        'Mixed Precision - scaler state_dict not found in loaded model. This may case issues '
                        'with loss scaling')
                else:
                    self.scaler.load_state_dict(scaler_state_dict)

    def _validate_final_average_model(self, cleanup_snapshots_pkl_file=False):
        """
        Testing the averaged model by loading the last saved average checkpoint and running test.
        Will be loaded to each of DDP processes
        :param cleanup_pkl_file: a flag for deleting the 10 best snapshots dictionary
        """
        logger.info('RUNNING ADDITIONAL TEST ON THE AVERAGED MODEL...')

        keep_state_dict = deepcopy(self.net.state_dict())
        # SETTING STATE DICT TO THE AVERAGE MODEL FOR EVALUATION
        average_model_ckpt_path = os.path.join(self.checkpoints_dir_path, self.average_model_checkpoint_filename)
        average_model_sd = read_ckpt_state_dict(average_model_ckpt_path)['net']

        self.net.load_state_dict(average_model_sd)
        # testing the averaged model and save instead of best model if needed
        averaged_model_results_tuple = self._validate_epoch(epoch=self.max_epochs)

        # Reverting the current model
        self.net.load_state_dict(keep_state_dict)

        if not self.ddp_silent_mode:
            # Adding values to sg_logger
            # looping over last titles which corresponds to validation (and average model) metrics.
            all_titles = self.results_titles[-1 * len(averaged_model_results_tuple):]
            result_dict = {all_titles[i]: averaged_model_results_tuple[i] for i in
                           range(len(averaged_model_results_tuple))}

            self.sg_logger.add_scalars(tag_scalar_dict=result_dict, global_step=self.max_epochs)

            average_model_tb_titles = ['Averaged Model ' + x for x in
                                       self.results_titles[-1 * len(averaged_model_results_tuple):]]
            write_struct = ''
            for ind, title in enumerate(average_model_tb_titles):
                write_struct += '%s: %.3f  \n  ' % (title, averaged_model_results_tuple[ind])
                self.sg_logger.add_scalar(title, averaged_model_results_tuple[ind], global_step=self.max_epochs)

            self.sg_logger.add_text("Averaged_Model_Performance", write_struct, self.max_epochs)
            if cleanup_snapshots_pkl_file:
                self.model_weight_averaging.cleanup()

    # FIXME - we need to resolve flake8's 'function is too complex' for this function
    @deprecated(version='0.1', reason="directly predict using the nn_module")  # noqa: C901
    def predict(self, inputs, targets=None, half=False, normalize=False, verbose=False,
                move_outputs_to_cpu=True):
        """
        A fast predictor for a batch of inputs
        :param inputs: torch.tensor or numpy.array
            a batch of inputs
        :param targets: torch.tensor()
            corresponding labels - if non are given - accuracy will not be computed
        :param verbose: bool
            print the results to screen
        :param normalize: bool
            If true, normalizes the tensor according to the dataloader's normalization values
        :param half:
            Performs half precision evaluation
        :param move_outputs_to_cpu:
            Moves the results from the GPU to the CPU
        :return: outputs, acc, net_time, gross_time
            networks predictions, accuracy calculation, forward pass net time, function gross time
        """

        transform_list = []

        # Create a 'to_tensor' transformation and a place holder of input_t
        if type(inputs) == torch.Tensor:
            inputs_t = torch.zeros_like(inputs)
        else:
            transform_list.append(transforms.ToTensor())
            inputs_t = torch.zeros(size=(inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2]))

        # Create a normalization transformation
        if normalize:
            try:
                mean, std = self.dataset_interface.lib_dataset_params['mean'], self.dataset_interface.lib_dataset_params['std']
            except AttributeError:
                raise AttributeError('In \'predict()\', Normalization is set to True while the dataset has no default '
                                     'mean & std => deactivate normalization or inject it to the datasets library.')
            transform_list.append(transforms.Normalize(mean, std))

        # Compose all transformations into one
        transformation = transforms.Compose(transform_list)

        # Transform the input
        for idx in range(len(inputs_t)):
            inputs_t[idx] = transformation(inputs[idx])

        # Timer instances
        gross_timer = core_utils.Timer('cpu')
        gross_timer.start()
        net_timer = core_utils.Timer(self.device)

        # Set network in eval mode
        self.net.eval()

        # Half is not supported on CPU
        if self.device != 'cuda' and half:
            half = False
            logger.warning('NOTICE: half is set to True but is not supported on CPU ==> using full precision')

        # Apply half precision to network and input
        if half:
            self.net.half()
            inputs_t = inputs_t.half()

        with torch.no_grad():
            # Move input to compute device
            inputs_t = inputs_t.to(self.device)

            # Forward pass (timed...)
            net_timer.start()
            outputs = self.net(inputs_t)
            net_time = net_timer.stop()

        if move_outputs_to_cpu:
            outputs = outputs.cpu()

        gross_time = gross_timer.stop()

        # Convert targets to tensor
        targets = torch.tensor(targets) if (type(targets) != torch.Tensor and targets is not None) else targets

        # Compute accuracy
        acc = metrics.accuracy(outputs.float(), targets.cpu())[0] if targets is not None else None
        acc_str = '%.2f' % acc if targets is not None else 'N/A'

        if verbose:
            logger.info('%s\nPredicted %d examples: \n\t%.2f ms (gross) --> %.2f ms (net)\n\tWith accuracy %s\n%s' %
                        ('-' * 50, inputs_t.shape[0], gross_time, net_time, acc_str, '-' * 50))

        # Undo the half precision
        if half and not self.half_precision:
            self.net = self.net.float()

        return outputs, acc, net_time, gross_time

    def compute_model_runtime(self, input_dims: tuple = None,
                              batch_sizes: Union[tuple, list, int] = (1, 8, 16, 32, 64),
                              verbose: bool = True):
        """
        Compute the "atomic" inference time and throughput.
        Atomic refers to calculating the forward pass independently, discarding effects such as data augmentation,
        data upload to device, multi-gpu distribution etc.
        :param input_dims: tuple
            shape of a basic input to the network (without the first index) e.g. (3, 224, 224)
            if None uses an input from the test loader
        :param batch_sizes: int or list
            Batch sizes for latency calculation
        :param verbose: bool
            Prints results to screen
        :return: log: dict
            Latency and throughput for each tested batch size
        """
        assert input_dims or self.test_loader is not None, 'Must get \'input_dims\' or connect a dataset interface'
        assert self.multi_gpu not in (MultiGPUMode.DATA_PARALLEL, MultiGPUMode.DISTRIBUTED_DATA_PARALLEL), \
            'The model is on multiple GPUs, move it to a single GPU is order to compute runtime'

        # TRANSFER THE MODEL TO EVALUATION MODE BUT REMEMBER THE MODE TO RETURN TO
        was_in_training_mode = True if self.net.training else False
        self.net.eval()

        # INITIALIZE LOGS AND PRINTS
        timer = core_utils.Timer(self.device)
        logs = {}
        log_print = f"{'-' * 35}\n" \
                    f"Batch   Time per Batch  Throughput\n" \
                    f"size         (ms)        (im/s)\n" \
                    f"{'-' * 35}\n"

        # GET THE INPUT SHAPE FROM THE DATA LOADER IF NOT PROVIDED EXPLICITLY
        input_dims = input_dims or next(iter(self.test_loader))[0].shape[1:]

        # DEFINE NUMBER ACCORDING TO DEVICE
        repetitions = 200 if self.device == 'cuda' else 20

        # CREATE A LIST OF BATCH SIZES
        batch_sizes = [batch_sizes] if type(batch_sizes) == int else batch_sizes

        for batch_size in sorted(batch_sizes):
            try:
                # CREATE A RANDOM TENSOR AS INPUT
                dummy_batch = torch.randn((batch_size, *input_dims), device=self.device)

                # WARM UP
                for _ in range(10):
                    _ = self.net(dummy_batch)

                # RUN & TIME
                accumulated_time = 0
                with torch.no_grad():
                    for _ in range(repetitions):
                        timer.start()
                        _ = self.net(dummy_batch)
                        accumulated_time += timer.stop()

                # PERFORMANCE CALCULATION
                time_per_batch = accumulated_time / repetitions
                throughput = batch_size * 1000 / time_per_batch

                logs[batch_size] = {'time_per_batch': time_per_batch, 'throughput': throughput}
                log_print += f"{batch_size:4.0f} {time_per_batch:12.1f} {throughput:12.0f}\n"

            except RuntimeError as e:
                # ONLY FOR THE CASE OF CUDA OUT OF MEMORY WE CATCH THE EXCEPTION AND CONTINUE THE FUNCTION
                if 'CUDA out of memory' in str(e):
                    log_print += f"{batch_size:4d}\t{'CUDA out of memory':13s}\n"
                else:
                    raise

        # PRINT RESULTS
        if verbose:
            logger.info(log_print)

        # RETURN THE MODEL TO THE PREVIOUS MODE
        self.net.train(was_in_training_mode)

        return logs

    def get_arch_params(self):
        return self.arch_params.to_dict()

    def get_structure(self):
        return self.net.module.structure

    def get_architecture(self):
        return self.architecture

    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name

    def re_build_model(self, arch_params={}):
        """
        arch_params : dict
            Architecture H.P. e.g.: block, num_blocks, num_classes, etc.
        :return:
        """
        if 'num_classes' not in arch_params.keys():
            if self.dataset_interface is None:
                raise Exception('Error', 'Number of classes not defined in arch params and dataset is not defined')
            else:
                arch_params['num_classes'] = len(self.classes)

        self.arch_params = core_utils.HpmStruct(**arch_params)
        self.classes = self.arch_params.num_classes
        self.net = self.architecture_cls(arch_params=self.arch_params)
        # save the architecture for neural architecture search
        if hasattr(self.net, 'structure'):
            self.architecture = self.net.structure

        self.net.to(self.device)

        if self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            logger.warning("Warning: distributed training is not supported in re_build_model()")
        self.net = torch.nn.DataParallel(self.net,
                                         device_ids=self.device_ids) if self.multi_gpu else core_utils.WrappedModel(
            self.net)

    def update_architecture(self, structure):
        '''
        architecture : str
            Defines the network's architecture according to the options in models/all_architectures
        load_checkpoint : bool
            Loads a checkpoint according to experiment_name
        arch_params : dict
            Architecture H.P. e.g.: block, num_blocks, num_classes, etc.
        :return:
        '''
        if hasattr(self.net.module, 'update_structure'):

            self.net.module.update_structure(structure)
            self.net.to(self.device)

        else:
            raise Exception("architecture is not valid for NAS")

    def get_module(self):
        return self.net

    def set_module(self, module):
        self.net = module

    def _initialize_device(self, requested_device: str, requested_multi_gpu: Union[MultiGPUMode, str]):
        """
        _initialize_device - Initializes the device for the model - Default is CUDA
            :param requested_device:        Device to initialize ('cuda' / 'cpu')
            :param requested_multi_gpu:     Get Multiple GPU
        """

        if isinstance(requested_multi_gpu, str):
            requested_multi_gpu = MultiGPUMode(requested_multi_gpu)

        # SELECT CUDA DEVICE
        if requested_device == 'cuda':
            if torch.cuda.is_available():
                self.device = 'cuda'  # TODO - we may want to set the device number as well i.e. 'cuda:1'
            else:
                raise RuntimeError('CUDA DEVICE NOT FOUND... EXITING')

        # SELECT CPU DEVICE
        elif requested_device == 'cpu':
            self.device = 'cpu'
            self.multi_gpu = False
        else:
            # SELECT CUDA DEVICE BY DEFAULT IF AVAILABLE
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # DEFUALT IS SET TO 1 - IT IS CHANGED IF MULTI-GPU IS USED
        self.num_devices = 1

        # IN CASE OF MULTIPLE GPUS UPDATE THE LEARNING AND DATA PARAMETERS
        # FIXME - CREATE A DISCUSSION ON THESE PARAMETERS - WE MIGHT WANT TO CHANGE THE WAY WE USE THE LR AND
        if requested_multi_gpu != MultiGPUMode.OFF:
            if 'cuda' in self.device:
                # COLLECT THE AVAILABLE GPU AND COUNT THE AVAILABLE GPUS AMOUNT
                self.device_ids = list(range(torch.cuda.device_count()))
                self.num_devices = len(self.device_ids)
                if self.num_devices == 1:
                    self.multi_gpu = MultiGPUMode.OFF
                    if requested_multi_gpu != MultiGPUMode.AUTO:
                        # if AUTO mode was set - do not log a warning
                        logger.warning('\n[WARNING] - Tried running on multiple GPU but only a single GPU is available\n')
                else:
                    if requested_multi_gpu == MultiGPUMode.AUTO:
                        if env_helpers.is_distributed():
                            requested_multi_gpu = MultiGPUMode.DISTRIBUTED_DATA_PARALLEL
                        else:
                            requested_multi_gpu = MultiGPUMode.DATA_PARALLEL

                    self.multi_gpu = requested_multi_gpu
                    if self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
                        self._initialize_ddp()
            else:
                # MULTIPLE GPUS CAN BE ACTIVE ONLY IF A GPU IS AVAILABLE
                self.multi_gpu = MultiGPUMode.OFF
                logger.warning('\n[WARNING] - Tried running on multiple GPU but none are available => running on CPU\n')

    def _initialize_ddp(self):
        """
        Initializes Distributed Data Parallel

        Usage:

            python -m torch.distributed.launch --nproc_per_node=n YOUR_TRAINING_SCRIPT.py
            where n is the number of GPUs required, e.g., n=8

            Important note: (1) in distributed training it is customary to specify learning rates and batch sizes per GPU.
            Whatever learning rate and schedule you specify will be applied to the each GPU individually.
            Since gradients are passed and summed (reduced) from all to all GPUs, the effective batch size is the
            batch you specify times the number of GPUs. In the literature there are several "best practices" to set
            learning rates and schedules for large batch sizes.

        """
        logger.info("Distributed training starting...")
        local_rank = environment_config.DDP_LOCAL_RANK
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if local_rank > 0:
            f = open(os.devnull, 'w')
            sys.stdout = f  # silent all printing for non master process

        torch.cuda.set_device(local_rank)
        self.device = 'cuda:%d' % local_rank

        # MAKE ALL HIGHER-RANK GPUS SILENT (DISTRIBUTED MODE)
        self.ddp_silent_mode = local_rank > 0

        if torch.distributed.get_rank() == 0:
            logger.info(f"Training in distributed mode... with {str(torch.distributed.get_world_size())} GPUs")

    def _switch_device(self, new_device):
        self.device = new_device
        self.net.to(self.device)

    # FIXME - we need to resolve flake8's 'function is too complex' for this function
    def _load_checkpoint_to_model(self, strict: StrictLoad, load_backbone: bool, source_ckpt_folder_name: str,
                                  load_ema_as_net: bool):  # noqa: C901 - too complex
        """
        Copies the source checkpoint to a local folder and loads the checkpoint's data to the model
        :param strict:           See StrictLoad class documentation for details.
        :param load_backbone:    loads the provided checkpoint to self.net.backbone instead of self.net
        :param source_ckpt_folder_name: The folder where the checkpoint is saved. By default uses the self.experiment_name
        NOTE: 'acc', 'epoch', 'optimizer_state_dict' and the logs are NOT loaded if self.zeroize_prev_train_params is True
        """

        # GET LOCAL PATH TO THE CHECKPOINT FILE FIRST
        ckpt_local_path = get_ckpt_local_path(source_ckpt_folder_name=source_ckpt_folder_name,
                                              experiment_name=self.experiment_name,
                                              ckpt_name=self.ckpt_name,
                                              model_checkpoints_location=self.model_checkpoints_location,
                                              external_checkpoint_path=self.external_checkpoint_path,
                                              overwrite_local_checkpoint=self.overwrite_local_checkpoint,
                                              load_weights_only=self.load_weights_only)

        # LOAD CHECKPOINT TO MODEL
        self.checkpoint = load_checkpoint_to_model(ckpt_local_path=ckpt_local_path,
                                                   load_backbone=load_backbone,
                                                   net=self.net,
                                                   strict=strict.value if isinstance(strict, StrictLoad) else strict,
                                                   load_weights_only=self.load_weights_only,
                                                   load_ema_as_net=load_ema_as_net)

        if 'ema_net' in self.checkpoint.keys():
            logger.warning("[WARNING] Main network has been loaded from checkpoint but EMA network exists as well. It "
                           " will only be loaded during validation when training with ema=True. ")

        # UPDATE TRAINING PARAMS IF THEY EXIST & WE ARE NOT LOADING AN EXTERNAL MODEL's WEIGHTS
        self.best_metric = self.checkpoint['acc'] if 'acc' in self.checkpoint.keys() else -1
        self.start_epoch = self.checkpoint['epoch'] if 'epoch' in self.checkpoint.keys() else 0

    def _prep_for_test(self, test_loader: torch.utils.data.DataLoader = None, loss=None, post_prediction_callback=None,
                       test_metrics_list=None,
                       loss_logging_items_names=None, test_phase_callbacks=None):
        """Run commands that are common to all SgModels"""
        # SET THE MODEL IN evaluation STATE
        self.net.eval()

        # IF SPECIFIED IN THE FUNCTION CALL - OVERRIDE THE self ARGUMENTS
        self.test_loader = test_loader or self.test_loader
        self.criterion = loss or self.criterion
        self.post_prediction_callback = post_prediction_callback or self.post_prediction_callback
        self.loss_logging_items_names = loss_logging_items_names or self.loss_logging_items_names
        self.phase_callbacks = test_phase_callbacks or self.phase_callbacks

        if self.phase_callbacks is None:
            self.phase_callbacks = []

        if test_metrics_list:
            self.test_metrics = MetricCollection(test_metrics_list)
            self.phase_callbacks.append(MetricsUpdateCallback(Phase.TEST_BATCH_END))
            self.phase_callback_handler = CallbackHandler(self.phase_callbacks)

        # WHEN TESTING WITHOUT A LOSS FUNCTION- CREATE EPOCH HEADERS FOR PRINTS
        if self.criterion is None:
            self.loss_logging_items_names = []

        if self.test_metrics is None:
            raise ValueError("Metrics are required to perform test. Pass them through test_metrics_list arg when "
                             "calling test or through training_params when calling train(...)")
        if self.test_loader is None:
            raise ValueError("Test dataloader is required to perform test. Make sure to either pass it through "
                             "test_loader arg or calling connect_dataset_interface upon a DatasetInterface instance "
                             "with a non empty testset attribute.")

        # RESET METRIC RUNNERS
        self.test_metrics.reset()
        self.test_metrics.to(self.device)

    def _initialize_sg_logger_objects(self):
        """Initialize object that collect, write to disk, monitor and store remotely all training outputs"""
        sg_logger = core_utils.get_param(self.training_params, 'sg_logger')

        # OVERRIDE SOME PARAMETERS TO MAKE SURE THEY MATCH THE TRAINING PARAMETERS
        general_sg_logger_params = {'experiment_name': self.experiment_name,
                                    'storage_location': self.model_checkpoints_location,
                                    'resumed': self.load_checkpoint,
                                    'training_params': self.training_params,
                                    'checkpoints_dir_path': self.checkpoints_dir_path}

        if sg_logger is None:
            raise RuntimeError('sg_logger must be defined in training params (see default_training_params)')

        if isinstance(sg_logger, AbstractSGLogger):
            self.sg_logger = sg_logger
        elif isinstance(sg_logger, str):
            sg_logger_params = core_utils.get_param(self.training_params, 'sg_logger_params', {})
            if issubclass(SG_LOGGERS[sg_logger], BaseSGLogger):
                sg_logger_params = {**sg_logger_params, **general_sg_logger_params}
            if sg_logger not in SG_LOGGERS:
                raise RuntimeError('sg_logger not defined in SG_LOGGERS')

            self.sg_logger = SG_LOGGERS[sg_logger](**sg_logger_params)
        else:
            raise RuntimeError('sg_logger can be either an sg_logger name (str) or a subcalss of AbstractSGLogger')

        if not isinstance(self.sg_logger, BaseSGLogger):
            logger.warning("WARNING! Using a user-defined sg_logger: files will not be automatically written to disk!\n"
                           "Please make sure the provided sg_logger writes to disk or compose your sg_logger to BaseSGLogger")

        # IN CASE SG_LOGGER UPDATED THE DIR PATH
        self.checkpoints_dir_path = self.sg_logger.local_dir()
        additional_log_items = {'initial_LR': self.training_params.initial_lr,
                                'num_devices': self.num_devices,
                                'multi_gpu': str(self.multi_gpu),
                                'device_type': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}

        # ADD INSTALLED PACKAGE LIST + THEIR VERSIONS
        if self.training_params.log_installed_packages:
            pkg_list = list(map(lambda pkg: str(pkg), _get_installed_distributions()))
            additional_log_items['installed_packages'] = pkg_list

        self.sg_logger.add_config("hyper_params", {"arch_params": self.arch_params.__dict__,
                                                   "training_hyperparams": self.training_params.__dict__,
                                                   "dataset_params": self.dataset_params.__dict__,
                                                   "additional_log_items": additional_log_items})

        self.sg_logger.flush()

    def _write_to_disk_operations(self, train_metrics: tuple, validation_results: tuple, inf_time: float, epoch: int, context: PhaseContext):
        """Run the various logging operations, e.g.: log file, Tensorboard, save checkpoint etc."""
        # STORE VALUES IN A TENSORBOARD FILE
        train_results = list(train_metrics) + list(validation_results) + [inf_time]
        all_titles = self.results_titles + ['Inference Time']

        result_dict = {all_titles[i]: train_results[i] for i in range(len(train_results))}
        self.sg_logger.add_scalars(tag_scalar_dict=result_dict, global_step=epoch)

        # SAVE THE CHECKPOINT
        if self.training_params.save_model:
            self.save_checkpoint(self.optimizer, epoch + 1, validation_results, context)

    def _write_lrs(self, epoch):
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        lr_titles = ['LR/Param_group_' + str(i) for i in range(len(self.optimizer.param_groups))] if len(
            self.optimizer.param_groups) > 1 else ['LR']
        lr_dict = {lr_titles[i]: lrs[i] for i in range(len(lrs))}
        self.sg_logger.add_scalars(tag_scalar_dict=lr_dict, global_step=epoch)

    def test(self,  # noqa: C901
             test_loader: torch.utils.data.DataLoader = None,
             loss: torch.nn.modules.loss._Loss = None,
             silent_mode: bool = False,
             test_metrics_list=None,
             loss_logging_items_names=None, metrics_progress_verbose=False, test_phase_callbacks=None,
             use_ema_net=True) -> tuple:
        """
        Evaluates the model on given dataloader and metrics.

        :param test_loader: dataloader to perform test on.
        :param test_metrics_list: (list(torchmetrics.Metric)) metrics list for evaluation.
        :param silent_mode: (bool) controls verbosity
        :param metrics_progress_verbose: (bool) controls the verbosity of metrics progress (default=False). Slows down the program.
        :param use_ema_net (bool) whether to perform test on self.ema_model.ema (when self.ema_model.ema exists,
            otherwise self.net will be tested) (default=True)
        :return: results tuple (tuple) containing the loss items and metric values.

        All of the above args will override SgModel's corresponding attribute when not equal to None. Then evaluation
         is ran on self.test_loader with self.test_metrics.
        """

        # IN CASE TRAINING WAS PERFROMED BEFORE TEST- MAKE SURE TO TEST THE EMA MODEL (UNLESS SPECIFIED OTHERWISE BY
        # use_ema_net)

        if use_ema_net and self.ema_model is not None:
            keep_model = self.net
            self.net = self.ema_model.ema

        self._prep_for_test(test_loader=test_loader,
                            loss=loss,
                            test_metrics_list=test_metrics_list,
                            loss_logging_items_names=loss_logging_items_names,
                            test_phase_callbacks=test_phase_callbacks,
                            )

        test_results = self.evaluate(data_loader=self.test_loader,
                                     metrics=self.test_metrics,
                                     evaluation_type=EvaluationType.TEST,
                                     silent_mode=silent_mode,
                                     metrics_progress_verbose=metrics_progress_verbose)

        # SWITCH BACK BETWEEN NETS SO AN ADDITIONAL TRAINING CAN BE DONE AFTER TEST
        if use_ema_net and self.ema_model is not None:
            self.net = keep_model

        return test_results

    def _validate_epoch(self, epoch: int, silent_mode: bool = False) -> tuple:
        """
        Runs evaluation on self.valid_loader, with self.valid_metrics.

        :param epoch: (int) epoch idx
        :param silent_mode: (bool) controls verbosity

        :return: results tuple (tuple) containing the loss items and metric values.
        """

        self.net.eval()
        self.valid_metrics.reset()
        self.valid_metrics.to(self.device)

        return self.evaluate(data_loader=self.valid_loader, metrics=self.valid_metrics,
                             evaluation_type=EvaluationType.VALIDATION, epoch=epoch, silent_mode=silent_mode)

    def evaluate(self, data_loader: torch.utils.data.DataLoader, metrics: MetricCollection,
                 evaluation_type: EvaluationType, epoch: int = None, silent_mode: bool = False,
                 metrics_progress_verbose: bool = False):
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
        progress_bar_data_loader = tqdm(data_loader, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True,
                                        disable=silent_mode)
        loss_avg_meter = core_utils.utils.AverageMeter()
        logging_values = None
        loss_tuple = None
        lr_warmup_epochs = self.training_params.lr_warmup_epochs if self.training_params else None
        context = PhaseContext(epoch=epoch,
                               metrics_compute_fn=metrics,
                               loss_avg_meter=loss_avg_meter,
                               criterion=self.criterion,
                               device=self.device,
                               lr_warmup_epochs=lr_warmup_epochs)

        if not silent_mode:
            # PRINT TITLES
            pbar_start_msg = f"Validation epoch {epoch}" if evaluation_type == EvaluationType.VALIDATION else "Test"
            progress_bar_data_loader.set_description(pbar_start_msg)

        with torch.no_grad():
            for batch_idx, batch_items in enumerate(progress_bar_data_loader):
                batch_items = core_utils.tensor_container_to_device(batch_items, self.device, non_blocking=True)
                inputs, targets, additional_batch_items = sg_model_utils.unpack_batch_items(batch_items)

                output = self.net(inputs)

                if self.criterion is not None:
                    # STORE THE loss_items ONLY, THE 1ST RETURNED VALUE IS THE loss FOR BACKPROP DURING TRAINING
                    loss_tuple = self._get_losses(output, targets)[1].cpu()

                context.update_context(batch_idx=batch_idx,
                                       inputs=inputs,
                                       preds=output,
                                       target=targets,
                                       loss_log_items=loss_tuple,
                                       **additional_batch_items)

                # TRIGGER PHASE CALLBACKS CORRESPONDING TO THE EVALUATION TYPE
                if evaluation_type == EvaluationType.VALIDATION:
                    self.phase_callback_handler(Phase.VALIDATION_BATCH_END, context)
                else:
                    self.phase_callback_handler(Phase.TEST_BATCH_END, context)

                # COMPUTE METRICS IF PROGRESS VERBOSITY IS SET
                if metrics_progress_verbose and not silent_mode:
                    # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
                    logging_values = get_logging_values(loss_avg_meter, metrics, self.criterion)
                    pbar_message_dict = get_train_loop_description_dict(logging_values,
                                                                        metrics,
                                                                        self.loss_logging_items_names)

                    progress_bar_data_loader.set_postfix(**pbar_message_dict)

        # NEED TO COMPUTE METRICS FOR THE FIRST TIME IF PROGRESS VERBOSITY IS NOT SET
        if not metrics_progress_verbose:
            # COMPUTE THE RUNNING USER METRICS AND LOSS RUNNING ITEMS. RESULT TUPLE IS THEIR CONCATENATION.
            logging_values = get_logging_values(loss_avg_meter, metrics, self.criterion)
            pbar_message_dict = get_train_loop_description_dict(logging_values,
                                                                metrics,
                                                                self.loss_logging_items_names)

            progress_bar_data_loader.set_postfix(**pbar_message_dict)

        # TODO: SUPPORT PRINTING AP PER CLASS- SINCE THE METRICS ARE NOT HARD CODED ANYMORE (as done in
        #  calc_batch_prediction_accuracy_per_class in metric_utils.py), THIS IS ONLY RELEVANT WHEN CHOOSING
        #  DETECTIONMETRICS, WHICH ALREADY RETURN THE METRICS VALUEST HEMSELVES AND NOT THE ITEMS REQUIRED FOR SUCH
        #  COMPUTATION. ALSO REMOVE THE BELOW LINES BY IMPLEMENTING CRITERION AS A TORCHMETRIC.

        if self.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            logging_values = reduce_results_tuple_for_ddp(logging_values, next(self.net.parameters()).device)
        return logging_values
