import copy
import csv
import itertools
import math
import numbers
import os
import signal
import time
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Sequence, Mapping

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.environment.checkpoints_dir_utils import get_project_checkpoints_dir_path
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.environment.device_utils import device_config
from super_gradients.common.factories.metrics_factory import MetricsFactory
from super_gradients.common.object_names import LRSchedulers, LRWarmups, Callbacks
from super_gradients.common.plugins.deci_client import DeciClient
from super_gradients.common.registry.registry import register_lr_scheduler, register_lr_warmup, register_callback, LR_SCHEDULERS_CLS_DICT, TORCH_LR_SCHEDULERS
from super_gradients.common.sg_loggers.time_units import GlobalBatchStepNumber, EpochNumber
from super_gradients.training.utils import get_param
from super_gradients.training.utils.callbacks.base_callbacks import PhaseCallback, PhaseContext, Phase, Callback
from super_gradients.training.utils.detection_utils import DetectionVisualization, DetectionPostPredictionCallback, cxcywh2xyxy, xyxy2cxcywh
from super_gradients.training.utils.distributed_training_utils import maybe_all_reduce_tensor_average, maybe_all_gather_as_list
from super_gradients.training.utils.segmentation_utils import BinarySegmentationVisualization
from super_gradients.training.utils.utils import unwrap_model, infer_model_device, tensor_container_to_device
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Metric
from torchvision.utils import draw_segmentation_masks

logger = get_logger(__name__)


@register_callback(Callbacks.MODEL_CONVERSION_CHECK)
class ModelConversionCheckCallback(PhaseCallback):
    """
    Pre-training callback that verifies model conversion to onnx given specified conversion parameters.

    The model is converted, then inference is applied with onnx runtime.

    Use this callback with the same args as DeciPlatformCallback to prevent conversion fails at the end of training.

    :param model_name:              Model's name
    :param input_dimensions:        Model's input dimensions
    :param primary_batch_size:      Model's primary batch size
    :param opset_version:           (default=11)
    :param do_constant_folding:     (default=True)
    :param dynamic_axes:            (default={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    :param input_names:             (default=["input"])
    :param output_names:            (default=["output"])
    :param rtol:                    (default=1e-03)
    :param atol:                    (default=1e-05)
    """

    def __init__(self, model_name: str, input_dimensions: Sequence[int], primary_batch_size: int, **kwargs):
        super(ModelConversionCheckCallback, self).__init__(phase=Phase.PRE_TRAINING)
        self.model_name = model_name
        self.input_dimensions = input_dimensions
        self.primary_batch_size = primary_batch_size

        self.opset_version = kwargs.get("opset_version", 10)
        self.do_constant_folding = kwargs.get("do_constant_folding", None) if kwargs.get("do_constant_folding", None) else True
        self.input_names = kwargs.get("input_names") or ["input"]
        self.output_names = kwargs.get("output_names") or ["output"]
        self.dynamic_axes = kwargs.get("dynamic_axes") or {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        self.rtol = kwargs.get("rtol", 1e-03)
        self.atol = kwargs.get("atol", 1e-05)

    def __call__(self, context: PhaseContext):
        model = copy.deepcopy(unwrap_model(context.net))
        model = model.cpu()
        model.eval()  # Put model into eval mode

        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(input_size=self.input_dimensions)

        x = torch.randn(self.primary_batch_size, *self.input_dimensions, requires_grad=False)

        tmp_model_path = os.path.join(context.ckpt_dir, self.model_name + "_tmp.onnx")

        with torch.no_grad():
            torch_out = model(x)

        torch.onnx.export(
            model,  # Model being run
            x,  # Model input (or a tuple for multiple inputs)
            tmp_model_path,  # Where to save the model (can be a file or file-like object)
            export_params=True,  # Store the trained parameter weights inside the model file
            opset_version=self.opset_version,
            do_constant_folding=self.do_constant_folding,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
        )

        onnx_model = onnx.load(tmp_model_path)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(tmp_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        # TODO: Ideally we don't want to check this but have the certainty of just calling torch_out.cpu()
        if isinstance(torch_out, List) or isinstance(torch_out, tuple):
            torch_out = torch_out[0]
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(torch_out.cpu().numpy(), ort_outs[0], rtol=self.rtol, atol=self.atol)

        os.remove(tmp_model_path)

        logger.info("Exported model has been tested with ONNXRuntime, and the result looks good!")


@register_callback(Callbacks.DECI_LAB_UPLOAD)
class DeciLabUploadCallback(PhaseCallback):
    """
    Post-training callback for uploading and optimizing a model.

    :param model_meta_data:             Model's meta-data object. Type: ModelMetadata
    :param optimization_request_form:   Optimization request form object. Type: OptimizationRequestForm
    :param ckpt_name:                   Checkpoint filename, inside the checkpoint directory.
    """

    def __init__(
        self,
        model_name: str,
        input_dimensions: Sequence[int],
        target_hardware_types: "Optional[List[str]]" = None,
        target_batch_size: "Optional[int]" = None,
        target_quantization_level: "Optional[str]" = None,
        ckpt_name: str = "ckpt_best.pth",
        **kwargs,
    ):
        super().__init__(phase=Phase.POST_TRAINING)
        self.input_dimensions = input_dimensions
        self.model_name = model_name
        self.target_hardware_types = target_hardware_types
        self.target_batch_size = target_batch_size
        self.target_quantization_level = target_quantization_level
        self.ckpt_name = ckpt_name
        self.platform_client = DeciClient()

    @staticmethod
    def log_optimization_failed():
        logger.info("We couldn't finish your model optimization. Visit https://console.deci.ai for details")

    def upload_model(self, model):
        """
        This function will upload the trained model to the Deci Lab

        :param model: The resulting model from the training process
        """
        self.platform_client.upload_model(
            model=model,
            name=self.model_name,
            input_dimensions=self.input_dimensions,
            target_hardware_types=self.target_hardware_types,
            target_batch_size=self.target_batch_size,
            target_quantization_level=self.target_quantization_level,
        )

    def get_optimization_status(self, optimized_model_name: str):
        """
        This function will do fetch the optimized version of the trained model and check on its benchmark status.
        The status will be checked against the server every 30 seconds and the process will timeout after 30 minutes
        or log about the successful optimization - whichever happens first.

        :param optimized_model_name: Optimized model name

        :return: Whether or not the optimized model has been benchmarked
        """

        def handler(_signum, _frame):
            logger.error("Process timed out. Visit https://console.deci.ai for details")
            return False

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(1800)

        finished = False
        while not finished:
            if self.platform_client.is_model_benchmarking(name=optimized_model_name):
                time.sleep(30)
            else:
                finished = True

        signal.alarm(0)
        return True

    def __call__(self, context: PhaseContext) -> None:
        """
        This function will attempt to upload the trained model and schedule an optimization for it.

        :param context: Training phase context
        """
        try:
            model = copy.deepcopy(unwrap_model(context.net))
            model_state_dict_path = os.path.join(context.ckpt_dir, self.ckpt_name)
            model_state_dict = torch.load(model_state_dict_path)["net"]
            model.load_state_dict(state_dict=model_state_dict)

            model = model.cpu()
            if hasattr(model, "prep_model_for_conversion"):
                model.prep_model_for_conversion(input_size=self.input_dimensions)

            self.upload_model(model=model)
            model_name = self.model_name
            logger.info(f"Successfully added {model_name} to the model repository")

            optimized_model_name = f"{model_name}_1_1"
            logger.info("We'll wait for the scheduled optimization to finish. Please don't close this window")
            success = self.get_optimization_status(optimized_model_name=optimized_model_name)
            if success:
                logger.info("Successfully finished your model optimization. Visit https://console.deci.ai for details")
            else:
                DeciLabUploadCallback.log_optimization_failed()
        except Exception as ex:
            DeciLabUploadCallback.log_optimization_failed()
            logger.error(ex)


@register_callback(Callbacks.LR_CALLBACK_BASE)
class LRCallbackBase(PhaseCallback):
    """
    Base class for hard coded learning rate scheduling regimes, implemented as callbacks.
    """

    def __init__(self, phase, initial_lr, update_param_groups, train_loader_len, net, training_params, **kwargs):
        super(LRCallbackBase, self).__init__(phase)
        if not isinstance(initial_lr, dict):
            initial_lr = {"default": float(initial_lr)}
        self.initial_lr = initial_lr
        self.lr = initial_lr.copy()
        self.update_param_groups = update_param_groups
        self.train_loader_len = train_loader_len
        self.net = net
        self.training_params = training_params

    def __call__(self, context: PhaseContext, **kwargs):
        if self.is_lr_scheduling_enabled(context):
            self.perform_scheduling(context)

    def is_lr_scheduling_enabled(self, context: PhaseContext):
        """
        Predicate that controls whether to perform lr scheduling based on values in context.

        :param context: PhaseContext: current phase's context.
        :return: bool, whether to apply lr scheduling or not.
        """
        raise NotImplementedError

    def perform_scheduling(self, context: PhaseContext):
        """
        Performs lr scheduling based on values in context.

        :param context: PhaseContext: current phase's context.
        """
        raise NotImplementedError

    def update_lr(self, optimizer, epoch, batch_idx=None):
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr[param_group["name"]]


@register_lr_warmup(LRWarmups.LINEAR_EPOCH_STEP, deprecated_name="linear_epoch_step")
class LinearEpochLRWarmup(LRCallbackBase):
    """
    LR scheduling callback for linear step warmup. This scheduler uses a whole epoch as single step.
    LR climbs from warmup_initial_lr with even steps to initial lr. When warmup_initial_lr is None - LR climb starts from
     initial_lr/(1+warmup_epochs).

    """

    def __init__(self, **kwargs):
        super().__init__(Phase.TRAIN_EPOCH_START, **kwargs)
        warmup_initial_lr = {}
        if self.training_params.warmup_initial_lr is not None:
            if isinstance(self.training_params.warmup_initial_lr, float):
                for group_name in self.initial_lr.keys():
                    warmup_initial_lr[group_name] = self.training_params.warmup_initial_lr
            elif isinstance(self.training_params.warmup_initial_lr, Mapping):
                warmup_initial_lr = self.training_params.warmup_initial_lr
            else:
                raise TypeError("Warmup initial lr expected to be of type float or Mapping.")
        else:
            for group_name in self.initial_lr.keys():
                warmup_initial_lr[group_name] = self.initial_lr[group_name] / (self.training_params.lr_warmup_epochs + 1)
        self.warmup_initial_lr = warmup_initial_lr

        warmup_step_size = {}
        for group_name in self.initial_lr.keys():
            warmup_step_size[group_name] = (
                (self.initial_lr[group_name] - self.warmup_initial_lr[group_name]) / self.training_params.lr_warmup_epochs
                if self.training_params.lr_warmup_epochs > 0
                else 0
            )
        self.warmup_step_size = warmup_step_size

    def perform_scheduling(self, context):
        for group_name in self.initial_lr.keys():
            self.lr[group_name] = self.warmup_initial_lr[group_name] + context.epoch * self.warmup_step_size[group_name]
        self.update_lr(context.optimizer, context.epoch, None)

    def is_lr_scheduling_enabled(self, context):
        return self.training_params.lr_warmup_epochs > 0 and self.training_params.lr_warmup_epochs >= context.epoch


@register_lr_warmup(LRWarmups.LINEAR_BATCH_STEP, deprecated_name="linear_batch_step")
class LinearBatchLRWarmup(Callback):
    """
    LR scheduling callback for linear step warmup on each batch step.
    LR climbs from warmup_initial_lr with to initial lr.
    """

    def __init__(
        self,
        warmup_initial_lr: float,
        initial_lr: float,
        train_loader_len: int,
        lr_warmup_steps: int,
        training_params,
        net,
        **kwargs,
    ):
        """

        :param warmup_initial_lr: Starting learning rate
        :param initial_lr: Target learning rate after warmup
        :param train_loader_len: Length of train data loader
        :param lr_warmup_steps: Optional. If passed, will use fixed number of warmup steps to warmup LR. Default is None.
        :param kwargs:
        """

        super().__init__()

        if lr_warmup_steps > train_loader_len:
            logger.warning(
                f"Number of warmup steps ({lr_warmup_steps}) is greater than number of steps in epoch ({train_loader_len}). "
                f"Warmup steps will be capped to number of steps in epoch to avoid interfering with any pre-epoch LR schedulers."
            )

        if isinstance(initial_lr, numbers.Number):
            initial_lr = {"default": initial_lr}
        self.initial_lr = initial_lr
        self.lr = initial_lr.copy()

        if isinstance(warmup_initial_lr, numbers.Number):
            warmup_initial_lr = {group_name: warmup_initial_lr for group_name in self.lr.keys()}
        elif isinstance(warmup_initial_lr, Mapping):
            warmup_initial_lr = warmup_initial_lr
        else:
            raise TypeError("Warmup initial lr expected to be of type float or Mapping.")

        lr_warmup_steps = min(lr_warmup_steps, train_loader_len)
        learning_rates = {
            group_name: np.linspace(start=warmup_initial_lr[group_name], stop=initial_lr[group_name], num=lr_warmup_steps, endpoint=True)
            for group_name in self.initial_lr.keys()
        }
        self.training_params = training_params
        self.net = net
        self.learning_rates = learning_rates
        self.train_loader_len = train_loader_len
        self.lr_warmup_steps = lr_warmup_steps

    def on_train_batch_start(self, context: PhaseContext) -> None:
        global_training_step = context.batch_idx + context.epoch * self.train_loader_len
        if global_training_step < self.lr_warmup_steps:
            for group_name in self.initial_lr.keys():
                self.lr[group_name] = float(self.learning_rates[group_name][global_training_step])
            self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def update_lr(self, optimizer, epoch, batch_idx=None):
        """
        Same as in LRCallbackBase
        :param optimizer:
        :param epoch:
        :param batch_idx:
        :return:
        """
        # UPDATE THE OPTIMIZERS PARAMETER
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr[param_group["name"]]


@register_lr_scheduler(LRSchedulers.STEP, deprecated_name="step")
class StepLRScheduler(LRCallbackBase):
    """
    Hard coded step learning rate scheduling (i.e at specific milestones).
    """

    def __init__(self, lr_updates, lr_decay_factor, step_lr_update_freq=None, **kwargs):
        super().__init__(Phase.TRAIN_EPOCH_END, **kwargs)
        if step_lr_update_freq and len(lr_updates):
            raise ValueError(
                "Parameters lr_updates and step_lr_update_freq are mutually exclusive"
                f" and cannot be passed to {StepLRScheduler.__name__} constructor simultaneously"
            )

        if step_lr_update_freq is None and len(lr_updates) == 0:
            raise ValueError(f"At least one of [lr_updates, step_lr_update_freq] parameters should be passed to {StepLRScheduler.__name__} constructor")

        if step_lr_update_freq:
            max_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
            warmup_epochs = self.training_params.lr_warmup_epochs
            lr_updates = [
                int(np.ceil(step_lr_update_freq * x)) for x in range(1, max_epochs) if warmup_epochs <= int(np.ceil(step_lr_update_freq * x)) < max_epochs
            ]
        elif self.training_params.lr_cooldown_epochs > 0:
            logger.warning("Specific lr_updates were passed along with cooldown_epochs > 0," " cooldown will have no effect.")
        self.lr_updates = lr_updates
        self.lr_decay_factor = lr_decay_factor

    def perform_scheduling(self, context):
        num_updates_passed = [x for x in self.lr_updates if x <= context.epoch]
        for group_name in self.lr.keys():
            self.lr[group_name] = self.initial_lr[group_name] * self.lr_decay_factor ** len(num_updates_passed)
        self.update_lr(context.optimizer, context.epoch, None)

    def is_lr_scheduling_enabled(self, context):
        return self.training_params.lr_warmup_epochs <= context.epoch


@register_lr_scheduler(LRSchedulers.EXP, deprecated_name="exp")
class ExponentialLRScheduler(LRCallbackBase):
    """
    Exponential decay learning rate scheduling. Decays the learning rate by `lr_decay_factor` every epoch.
    """

    def __init__(self, lr_decay_factor: float, **kwargs):
        super().__init__(phase=Phase.TRAIN_BATCH_STEP, **kwargs)
        self.lr_decay_factor = lr_decay_factor

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        current_iter = self.train_loader_len * effective_epoch + context.batch_idx
        for group_name in self.lr.keys():
            self.lr[group_name] = self.initial_lr[group_name] * self.lr_decay_factor ** (current_iter / self.train_loader_len)
        self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def is_lr_scheduling_enabled(self, context):
        post_warmup_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        return self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs


@register_lr_scheduler(LRSchedulers.POLY, deprecated_name="poly")
class PolyLRScheduler(LRCallbackBase):
    """
    Hard coded polynomial decay learning rate scheduling (i.e at specific milestones).
    """

    def __init__(self, max_epochs, **kwargs):
        super().__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        self.max_epochs = max_epochs

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs - self.training_params.lr_cooldown_epochs
        current_iter = (self.train_loader_len * effective_epoch + context.batch_idx) / self.training_params.batch_accumulate
        max_iter = self.train_loader_len * effective_max_epochs / self.training_params.batch_accumulate
        for group_name in self.lr.keys():
            self.lr[group_name] = self.initial_lr[group_name] * pow((1.0 - (current_iter / max_iter)), 0.9)
        self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def is_lr_scheduling_enabled(self, context):
        post_warmup_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        return self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs


@register_lr_scheduler(LRSchedulers.COSINE, deprecated_name="cosine")
class CosineLRScheduler(LRCallbackBase):
    """
    Hard coded step Cosine anealing learning rate scheduling.
    """

    def __init__(self, max_epochs, cosine_final_lr_ratio, **kwargs):
        super().__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        self.max_epochs = max_epochs
        self.cosine_final_lr_ratio = cosine_final_lr_ratio

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs - self.training_params.lr_cooldown_epochs
        current_iter = max(0, self.train_loader_len * effective_epoch + context.batch_idx - self.training_params.lr_warmup_steps)
        max_iter = self.train_loader_len * effective_max_epochs - self.training_params.lr_warmup_steps
        for group_name in self.lr.keys():
            self.lr[group_name] = float(self.compute_learning_rate(current_iter, max_iter, self.initial_lr[group_name], self.cosine_final_lr_ratio))

        self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def is_lr_scheduling_enabled(self, context):
        # Account of per-step warmup
        if self.training_params.lr_warmup_steps > 0:
            current_step = self.train_loader_len * context.epoch + context.batch_idx
            return current_step >= self.training_params.lr_warmup_steps

        post_warmup_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        return self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs

    @classmethod
    def compute_learning_rate(cls, step: Union[float, np.ndarray], total_steps: float, initial_lr: float, final_lr_ratio: float):
        # the cosine starts from initial_lr and reaches initial_lr * cosine_final_lr_ratio in last epoch

        lr = 0.5 * initial_lr * (1.0 + np.cos(step / (total_steps + 1) * math.pi))
        return lr * (1 - final_lr_ratio) + (initial_lr * final_lr_ratio)


@register_lr_scheduler(LRSchedulers.FUNCTION, deprecated_name="function")
class FunctionLRScheduler(LRCallbackBase):
    """
    Hard coded rate scheduling for user defined lr scheduling function.
    """

    def __init__(self, max_epochs, lr_schedule_function, **kwargs):
        super().__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        assert callable(lr_schedule_function), "self.lr_function must be callable"
        self.lr_schedule_function = lr_schedule_function
        self.max_epochs = max_epochs

    def is_lr_scheduling_enabled(self, context):
        post_warmup_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        return self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs - self.training_params.lr_cooldown_epochs
        for group_name in self.lr.keys():
            self.lr[group_name] = self.lr_schedule_function(
                initial_lr=self.initial_lr[group_name],
                epoch=effective_epoch,
                iter=context.batch_idx,
                max_epoch=effective_max_epochs,
                iters_per_epoch=self.train_loader_len,
            )
        self.update_lr(context.optimizer, context.epoch, context.batch_idx)


class IllegalLRSchedulerMetric(Exception):
    """Exception raised illegal combination of training parameters.

    :param metric_name: Name of the metric that is not supported.
    :param metrics_dict: Dictionary of metrics that are supported.
    """

    def __init__(self, metric_name: str, metrics_dict: dict):
        self.message = "Illegal metric name: " + metric_name + ". Expected one of metics_dics keys: " + str(metrics_dict.keys())
        super().__init__(self.message)


@register_callback(Callbacks.LR_SCHEDULER)
class LRSchedulerCallback(PhaseCallback):
    """
    Learning rate scheduler callback.

    When passing __call__ a metrics_dict, with a key=self.metric_name, the value of that metric will monitored
         for ReduceLROnPlateau (i.e step(metrics_dict[self.metric_name]).

    :param scheduler:       Learning rate scheduler to be called step() with.
    :param metric_name:     Metric name for ReduceLROnPlateau learning rate scheduler.
    :param phase:           Phase of when to trigger it.
    """

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, phase: Union[Phase, str], metric_name: str = None):
        super(LRSchedulerCallback, self).__init__(phase)
        self.scheduler = scheduler
        self.metric_name = metric_name

    def __call__(self, context: PhaseContext):
        if context.lr_warmup_epochs <= context.epoch:
            if self.metric_name and self.metric_name in context.metrics_dict.keys():
                self.scheduler.step(context.metrics_dict[self.metric_name])
            elif self.metric_name is None:
                self.scheduler.step()
            else:
                raise IllegalLRSchedulerMetric(self.metric_name, context.metrics_dict)

    def __repr__(self):
        return "LRSchedulerCallback: " + repr(self.scheduler)


@register_callback(Callbacks.METRICS_UPDATE)
class MetricsUpdateCallback(PhaseCallback):
    def __init__(self, phase: Union[Phase, str]):
        super(MetricsUpdateCallback, self).__init__(phase)

    def __call__(self, context: PhaseContext):
        context.metrics_compute_fn.update(**context.__dict__)
        if context.criterion is not None:
            context.loss_avg_meter.update(context.loss_log_items, len(context.inputs))


class KDModelMetricsUpdateCallback(MetricsUpdateCallback):
    def __init__(self, phase: Union[Phase, str]):
        super().__init__(phase=phase)

    def __call__(self, context: PhaseContext):
        metrics_compute_fn_kwargs = {k: v.student_output if k == "preds" else v for k, v in context.__dict__.items()}
        context.metrics_compute_fn.update(**metrics_compute_fn_kwargs)
        if context.criterion is not None:
            context.loss_avg_meter.update(context.loss_log_items, len(context.inputs))


class PhaseContextTestCallback(PhaseCallback):
    """
    A callback that saves the phase context the for testing.
    """

    def __init__(self, phase: Union[Phase, str]):
        super(PhaseContextTestCallback, self).__init__(phase)
        self.context = None

    def __call__(self, context: PhaseContext):
        self.context = context


@register_callback(Callbacks.DETECTION_VISUALIZATION_CALLBACK)
class DetectionVisualizationCallback(PhaseCallback):
    """
    A callback that adds a visualization of a batch of detection predictions to context.sg_logger

    :param phase:                   When to trigger the callback.
    :param freq:                    Frequency (in epochs) to perform this callback.
    :param batch_idx:               Batch index to perform visualization for.
    :param classes:                 Class list of the dataset.
    :param last_img_idx_in_batch:   Last image index to add to log. (default=-1, will take entire batch).
    """

    def __init__(
        self,
        phase: Union[Phase, str],
        freq: int,
        post_prediction_callback: DetectionPostPredictionCallback,
        classes: list,
        batch_idx: int = 0,
        last_img_idx_in_batch: int = -1,
    ):
        super(DetectionVisualizationCallback, self).__init__(phase)
        self.freq = freq
        self.post_prediction_callback = post_prediction_callback
        self.batch_idx = batch_idx
        self.classes = classes
        self.last_img_idx_in_batch = last_img_idx_in_batch

    def __call__(self, context: PhaseContext):
        if context.epoch % self.freq == 0 and context.batch_idx == self.batch_idx and not context.ddp_silent_mode:
            # SOME CALCULATIONS ARE IN-PLACE IN NMS, SO CLONE THE PREDICTIONS
            preds = (context.preds[0].clone(), None)
            preds = self.post_prediction_callback(preds)
            batch_imgs = DetectionVisualization.visualize_batch(context.inputs, preds, context.target, self.batch_idx, self.classes)
            batch_imgs = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in batch_imgs]
            batch_imgs = np.stack(batch_imgs)
            tag = "batch_" + str(self.batch_idx) + "_images"
            context.sg_logger.add_images(tag=tag, images=batch_imgs[: self.last_img_idx_in_batch], global_step=context.epoch, data_format="NHWC")


class BinarySegmentationVisualizationCallback(PhaseCallback):
    """
    A callback that adds a visualization of a batch of segmentation predictions to context.sg_logger

    :param phase:                   When to trigger the callback.
    :param freq:                    Frequency (in epochs) to perform this callback.
    :param batch_idx:               Batch index to perform visualization for.
    :param last_img_idx_in_batch:   Last image index to add to log. (default=-1, will take entire batch).
    """

    def __init__(self, phase: Union[Phase, str], freq: int, batch_idx: int = 0, last_img_idx_in_batch: int = -1):
        super(BinarySegmentationVisualizationCallback, self).__init__(phase)
        self.freq = freq
        self.batch_idx = batch_idx
        self.last_img_idx_in_batch = last_img_idx_in_batch

    def __call__(self, context: PhaseContext):
        if context.epoch % self.freq == 0 and context.batch_idx == self.batch_idx:
            if isinstance(context.preds, tuple):
                preds = context.preds[0].clone()
            else:
                preds = context.preds.clone()
            batch_imgs = BinarySegmentationVisualization.visualize_batch(context.inputs, preds, context.target, self.batch_idx)
            batch_imgs = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in batch_imgs]
            batch_imgs = np.stack(batch_imgs)
            tag = "batch_" + str(self.batch_idx) + "_images"
            context.sg_logger.add_images(tag=tag, images=batch_imgs[: self.last_img_idx_in_batch], global_step=context.epoch, data_format="NHWC")


class TrainingStageSwitchCallbackBase(PhaseCallback):
    """
    TrainingStageSwitchCallback

    A phase callback that is called at a specific epoch (epoch start) to support multi-stage training.
    It does so by manipulating the objects inside the context.

    :param next_stage_start_epoch: Epoch idx to apply the stage change.
    """

    def __init__(self, next_stage_start_epoch: int):
        super(TrainingStageSwitchCallbackBase, self).__init__(phase=Phase.TRAIN_EPOCH_START)
        self.next_stage_start_epoch = next_stage_start_epoch

    def __call__(self, context: PhaseContext):
        if context.epoch == self.next_stage_start_epoch:
            self.apply_stage_change(context)

    def apply_stage_change(self, context: PhaseContext):
        """
        This method is called when the callback is fired on the next_stage_start_epoch,
         and holds the stage change logic that should be applied to the context's objects.

        :param context: PhaseContext, context of current phase
        """
        raise NotImplementedError


@register_callback(Callbacks.YOLOX_TRAINING_STAGE_SWITCH)
class YoloXTrainingStageSwitchCallback(TrainingStageSwitchCallbackBase):
    """
    YoloXTrainingStageSwitchCallback

    Training stage switch for YoloX training.
    Disables mosaic, and manipulates YoloX loss to use L1.

    """

    def __init__(self, next_stage_start_epoch: int = 285):
        super(YoloXTrainingStageSwitchCallback, self).__init__(next_stage_start_epoch=next_stage_start_epoch)

    def apply_stage_change(self, context: PhaseContext):
        for transform in context.train_loader.dataset.transforms:
            if hasattr(transform, "close"):
                transform.close()
        iter(context.train_loader)
        context.criterion.use_l1 = True


@register_callback(Callbacks.ROBOFLOW_RESULT_CALLBACK)
class RoboflowResultCallback(Callback):
    """Append the training results to a csv file. Be aware that this does not fully overwrite the existing file, just appends."""

    def __init__(self, dataset_name: str, output_path: Optional[str] = None):
        """
        :param dataset_name:    Name of the dataset that was used to train the model.
        :param output_path:     Full path to the output csv file. By default, save at 'checkpoint_dir/results.csv'
        """
        self.dataset_name = dataset_name
        self.output_path = output_path or os.path.join(get_project_checkpoints_dir_path(), "results.csv")

        if self.output_path is None:
            raise ValueError("Output path must be specified")

        super(RoboflowResultCallback, self).__init__()

    @multi_process_safe
    def on_training_end(self, context: PhaseContext):
        with open(self.output_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            mAP = context.metrics_dict["mAP@0.50:0.95"].item()
            writer.writerow([self.dataset_name, mAP])


class TestLRCallback(PhaseCallback):
    """
    Phase callback that collects the learning rates in lr_placeholder at the end of each epoch (used for testing). In
     the case of multiple parameter groups (i.e multiple learning rates) the learning rate is collected from the first
     one. The phase is VALIDATION_EPOCH_END to ensure all lr updates have been performed before calling this callback.
    """

    def __init__(self, lr_placeholder):
        super(TestLRCallback, self).__init__(Phase.VALIDATION_EPOCH_END)
        self.lr_placeholder = lr_placeholder

    def __call__(self, context: PhaseContext):
        self.lr_placeholder.append(context.optimizer.param_groups[0]["lr"])


@register_callback(Callbacks.TIMER)
class TimerCallback(Callback):
    def __init__(self):
        self.events = {}

    @multi_process_safe
    def on_train_loader_start(self, context: PhaseContext) -> None:
        self.events["on_train_loader_start"] = cv2.getTickCount()

    @multi_process_safe
    def on_train_batch_start(self, context: PhaseContext) -> None:
        self.events["on_train_batch_start"] = cv2.getTickCount()

    @multi_process_safe
    def on_train_batch_loss_end(self, context: PhaseContext) -> None:
        self.events["on_train_batch_loss_end"] = cv2.getTickCount()
        context.sg_logger.add_scalar(
            tag="timer/train_batch_forward_with_loss_ms",
            scalar_value=self._elapsed_time_between("on_train_batch_start", "on_train_batch_loss_end"),
            global_step=GlobalBatchStepNumber(self._infer_global_step(context, is_train_loader=True)),
        )

    @multi_process_safe
    def on_train_batch_gradient_step_start(self, context: PhaseContext) -> None:
        self.events["on_train_batch_gradient_step_start"] = cv2.getTickCount()

    @multi_process_safe
    def on_train_batch_gradient_step_end(self, context: PhaseContext) -> None:
        self.events["on_train_batch_gradient_step_end"] = cv2.getTickCount()
        context.sg_logger.add_scalar(
            tag="timer/train_batch_gradient_time",
            scalar_value=self._elapsed_time_between("on_train_batch_gradient_step_start", "on_train_batch_gradient_step_end"),
            global_step=GlobalBatchStepNumber(self._infer_global_step(context, is_train_loader=True)),
        )

    @multi_process_safe
    def on_train_batch_end(self, context: PhaseContext) -> None:
        self.events["on_train_batch_end"] = cv2.getTickCount()
        context.sg_logger.add_scalar(
            tag="timer/train_batch_total_time_ms",
            scalar_value=self._elapsed_time_between("on_train_batch_start", "on_train_batch_end"),
            global_step=GlobalBatchStepNumber(self._infer_global_step(context, is_train_loader=True)),
        )

    @multi_process_safe
    def on_train_loader_end(self, context: PhaseContext) -> None:
        self.events["on_train_loader_end"] = cv2.getTickCount()
        context.sg_logger.add_scalar(
            tag="timer/train_loader_total_time_ms",
            scalar_value=self._elapsed_time_between("on_train_loader_start", "on_train_loader_end"),
            global_step=EpochNumber(context.epoch),
        )

    @multi_process_safe
    def on_validation_loader_start(self, context: PhaseContext) -> None:
        self.events["on_validation_loader_start"] = cv2.getTickCount()

    @multi_process_safe
    def on_validation_batch_start(self, context: PhaseContext) -> None:
        self.events["on_validation_batch_start"] = cv2.getTickCount()

    @multi_process_safe
    def on_validation_batch_end(self, context: PhaseContext) -> None:
        self.events["on_validation_batch_end"] = cv2.getTickCount()
        context.sg_logger.add_scalar(
            tag="timer/validation_batch_total_time_ms",
            scalar_value=self._elapsed_time_between("on_validation_batch_start", "on_validation_batch_end"),
            global_step=GlobalBatchStepNumber(self._infer_global_step(context, is_train_loader=False)),
        )

    @multi_process_safe
    def on_validation_loader_end(self, context: PhaseContext) -> None:
        self.events["on_validation_loader_end"] = cv2.getTickCount()
        context.sg_logger.add_scalar(
            tag="timer/validation_loader_total_time_ms",
            scalar_value=self._elapsed_time_between("on_validation_loader_start", "on_validation_loader_end"),
            global_step=EpochNumber(context.epoch),
        )

        context.sg_logger.add_scalar(
            tag="timer/epoch_total_time_sec",
            scalar_value=self._elapsed_time_between("on_train_loader_start", "on_validation_loader_end") / 1000.0,
            global_step=EpochNumber(context.epoch),
        )

    def _elapsed_time_between(self, start_event, end_event):
        return 1000.0 * (self.events[end_event] - self.events[start_event]) / cv2.getTickFrequency()

    def _infer_global_step(self, context: PhaseContext, is_train_loader: bool):
        train_loader_length = len(context.train_loader) if context.train_loader is not None else 0
        valid_loader_length = len(context.valid_loader) if context.valid_loader is not None else 0
        total_steps_in_epoch = train_loader_length + valid_loader_length
        total_steps_in_done = context.epoch * total_steps_in_epoch
        if is_train_loader:
            return total_steps_in_done + context.batch_idx
        else:
            return total_steps_in_done + train_loader_length + context.batch_idx


@register_callback(Callbacks.SLIDING_WINDOW_VALIDATION)
class SlidingWindowValidationCallback(Callback):
    """
    Performing single-scale sliding window during inference at the last epoch on the validation set and on the average model.
    """

    def __init__(self, transforms_for_sliding_window) -> None:
        self.transforms_for_sliding_window = transforms_for_sliding_window
        self.valid_loader_transforms = []
        self.test_loader_transforms = []

    def on_validation_loader_start(self, context: PhaseContext) -> None:
        if context.training_params.max_epochs - 1 == context.epoch:
            unwrap_model(context.net).enable_sliding_window_validation()
            self.valid_loader_transforms = context.valid_loader.dataset.transforms.transforms
            context.valid_loader.dataset.transforms.transforms = self.transforms_for_sliding_window
            iter(context.valid_loader)

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        if context.training_params.max_epochs - 1 == context.epoch:
            unwrap_model(context.net).disable_sliding_window_validation()

    def on_average_best_models_validation_start(self, context: PhaseContext) -> None:
        if context.training_params.max_epochs - 1 == context.epoch and context.training_params.average_best_models:
            unwrap_model(context.net).enable_sliding_window_validation()
            context.valid_loader.dataset.transforms.transforms = self.transforms_for_sliding_window
            iter(context.valid_loader)

    def on_average_best_models_validation_end(self, context: PhaseContext) -> None:
        if context.training_params.max_epochs == context.epoch and context.training_params.average_best_models:
            unwrap_model(context.net).disable_sliding_window_validation()
            context.valid_loader.dataset.transforms.transforms = self.valid_loader_transforms
            iter(context.valid_loader)

    def on_test_loader_start(self, context: PhaseContext) -> None:
        unwrap_model(context.net).enable_sliding_window_validation()
        self.test_loader_transforms = context.test_loader.dataset.transforms.transforms
        context.test_loader.dataset.transforms.transforms = self.transforms_for_sliding_window
        iter(context.test_loader)

    def on_test_loader_end(self, context: PhaseContext) -> None:
        unwrap_model(context.net).disable_sliding_window_validation()
        context.test_loader.dataset.transforms.transforms = self.test_loader_transforms
        iter(context.test_loader)


def create_lr_scheduler_callback(
    lr_mode: Union[str, Mapping],
    train_loader: DataLoader,
    net: torch.nn.Module,
    training_params: Mapping,
    update_param_groups: bool,
    optimizer: torch.optim.Optimizer,
) -> PhaseCallback:
    """
    Creates the phase callback in charge of LR scheduling, to be used by Trainer.

    :param lr_mode: Union[str, Mapping],

                    When str:

                    Learning rate scheduling policy, one of ['StepLRScheduler','PolyLRScheduler','CosineLRScheduler','FunctionLRScheduler'].

                    'StepLRScheduler' refers to constant updates at epoch numbers passed through `lr_updates`.
                        Each update decays the learning rate by `lr_decay_factor`.

                    'CosineLRScheduler' refers to the Cosine Anealing policy as mentioned in https://arxiv.org/abs/1608.03983.
                      The final learning rate ratio is controlled by `cosine_final_lr_ratio` training parameter.

                    'PolyLRScheduler' refers to the polynomial decrease:
                        in each epoch iteration `self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)), 0.9)`

                    'FunctionLRScheduler' refers to a user-defined learning rate scheduling function, that is passed through `lr_schedule_function`.



                    When Mapping, refers to a torch.optim.lr_scheduler._LRScheduler, following the below API:

                        lr_mode = {LR_SCHEDULER_CLASS_NAME: {**LR_SCHEDULER_KWARGS, "phase": XXX, "metric_name": XXX)

                        Where "phase" (of Phase type) controls when to call torch.optim.lr_scheduler._LRScheduler.step().

                        For instance, in order to:
                        - Update LR on each batch: Use phase: Phase.TRAIN_BATCH_END
                        - Update LR after each epoch: Use phase: Phase.TRAIN_EPOCH_END

                        The "metric_name" refers to the metric to watch (See docs for "metric_to_watch" in train(...)
                         https://docs.deci.ai/super-gradients/docstring/training/sg_trainer.html) when using
                          ReduceLROnPlateau. In any other case this kwarg is ignored.

                        **LR_SCHEDULER_KWARGS are simply passed to the torch scheduler's __init__.




    :param train_loader: DataLoader, the Trainer.train_loader used for training.

    :param net: torch.nn.Module, the Trainer.net used for training.

    :param training_params: Mapping, Trainer.training_params.

    :param update_param_groups:bool,  Whether the Trainer.net has a specific way of updaitng its parameter group.

    :param optimizer: The optimizer used for training. Will be passed to the LR callback's __init__
     (or the torch scheduler's init, depending on the lr_mode value as described above).

    :return: a PhaseCallback instance to be used by Trainer for LR scheduling.
    """

    if isinstance(lr_mode, str) and lr_mode in LR_SCHEDULERS_CLS_DICT:
        sg_lr_callback_cls = LR_SCHEDULERS_CLS_DICT[lr_mode]
        sg_lr_callback = sg_lr_callback_cls(
            train_loader_len=len(train_loader),
            net=net,
            training_params=training_params,
            update_param_groups=update_param_groups,
            **training_params.to_dict(),
        )
    elif isinstance(lr_mode, Mapping) and list(lr_mode.keys())[0] in TORCH_LR_SCHEDULERS:
        if update_param_groups:
            logger.warning(
                "The network's way of updataing (i.e update_param_groups) is not supported with native " "torch lr schedulers and will have no effect."
            )
        lr_scheduler_name = list(lr_mode.keys())[0]
        torch_scheduler_params = {k: v for k, v in lr_mode[lr_scheduler_name].items() if k != "phase" and k != "metric_name"}
        torch_scheduler_params["optimizer"] = optimizer
        torch_scheduler = TORCH_LR_SCHEDULERS[lr_scheduler_name](**torch_scheduler_params)
        if get_param(lr_mode[lr_scheduler_name], "phase") is None:
            raise ValueError("Phase is required argument when working with torch schedulers.")

        if lr_scheduler_name == "ReduceLROnPlateau" and get_param(lr_mode[lr_scheduler_name], "metric_name") is None:
            raise ValueError("metric_name is required argument when working with ReduceLROnPlateau schedulers.")

        sg_lr_callback = LRSchedulerCallback(
            scheduler=torch_scheduler, phase=lr_mode[lr_scheduler_name]["phase"], metric_name=get_param(lr_mode[lr_scheduler_name], "metric_name")
        )
    else:
        raise ValueError(f"Unknown lr_mode: {lr_mode}")

    return sg_lr_callback


class ExtremeBatchCaseVisualizationCallback(Callback, ABC):
    """
    ExtremeBatchCaseVisualizationCallback

    A base class for visualizing worst/best validation batches in an epoch
     according to some metric or loss value, with Full DDP support.

    Images are saved with training_hyperparams.sg_logger.

    :param metric: Metric, will be the metric which is monitored.

    :param metric_component_name: In case metric returns multiple values (as Mapping),
     the value at metric.compute()[metric_component_name] will be the one monitored.

    :param loss_to_monitor: str, loss_to_monitor corresponfing to the 'criterion' passed through training_params in Trainer.train(...).
     Monitoring loss follows the same logic as metric_to_watch in Trainer.train(..), when watching the loss and should be:

        if hasattr(criterion, "component_names") and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"<COMPONENT_NAME>.

        If a single item is returned rather then a tuple:
            <LOSS_CLASS.__name__>.

        When there is no such attributesand criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"Loss_"<IDX>

    :param max: bool, Whether to take the batch corresponding to the max value of the metric/loss or
     the minimum (default=False).

    :param freq: int, epoch frequency to perform all of the above (default=1).

     Inheritors should implement process_extreme_batch which returns an image, as np.ndarray (uint8) with shape BHWC.
    """

    @resolve_param("metric", MetricsFactory())
    def __init__(
        self,
        metric: Optional[Metric] = None,
        metric_component_name: Optional[str] = None,
        loss_to_monitor: Optional[str] = None,
        max: bool = False,
        freq: int = 1,
        enable_on_train_loader: bool = False,
        enable_on_valid_loader: bool = True,
        max_images: int = -1,
    ):
        """
        :param metric: Metric, will be the metric which is monitored.

        :param metric_component_name: In case metric returns multiple values (as Mapping),
         the value at metric.compute()[metric_component_name] will be the one monitored.

        :param loss_to_monitor: str, loss_to_monitor corresponding to the 'criterion' passed through training_params in Trainer.train(...).
         Monitoring loss follows the same logic as metric_to_watch in Trainer.train(..), when watching the loss and should be:

        if hasattr(criterion, "component_names") and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"<COMPONENT_NAME>.

        If a single item is returned rather then a tuple:
            <LOSS_CLASS.__name__>.

        When there is no such attributes and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"Loss_"<IDX>

        :param max:                    bool, Whether to take the batch corresponding to the max value of the metric/loss or
        the minimum (default=False).

        :param freq:                   int, epoch frequency to perform all of the above (default=1).
        :param enable_on_train_loader: Controls whether to enable this callback on the train loader. Default is False.
        :param enable_on_valid_loader: Controls whether to enable this callback on the valid loader. Default is True.
        :param max_images:             Maximum images to save. If -1, save all images.
        """
        super(ExtremeBatchCaseVisualizationCallback, self).__init__()

        if (metric and loss_to_monitor) or (metric is None and loss_to_monitor is None):
            raise RuntimeError("Must pass exactly one of: loss, metric != None")

        self._set_tag_attr(loss_to_monitor, max, metric, metric_component_name)
        self.metric = metric
        if self.metric:
            self.metric = MetricCollection(self.metric)
            self.metric.to(device_config.device)

        self.metric_component_name = metric_component_name

        self.loss_to_monitor = loss_to_monitor
        self.max = max
        self.freq = freq

        self.extreme_score = None
        self.extreme_batch = None
        self.extreme_preds = None
        self.extreme_targets = None
        self.extreme_additional_batch_items = None

        self._first_call = True
        self._idx_loss_tuple = None

        self.enable_on_train_loader = enable_on_train_loader
        self.enable_on_valid_loader = enable_on_valid_loader
        self.max_images = max_images

    def _set_tag_attr(self, loss_to_monitor, max, metric, metric_component_name):
        if metric_component_name:
            monitored_val_name = metric_component_name
        elif metric:
            monitored_val_name = metric.__class__.__name__
        else:
            monitored_val_name = loss_to_monitor
        self._tag = f"max_{monitored_val_name}_batch" if max else f"min_{monitored_val_name}_batch"

    @abstractmethod
    def process_extreme_batch(self) -> np.ndarray:
        """
        This method is called right before adding the images to the in  SGLoggger (inside the on_validation_loader_end call).
         It should process self.extreme_batch, self.extreme_preds and self.extreme_targets and output the images, as np.ndarrray.
         Output should be of shape N,H,W,3 and uint8.
        :return: images to save, np.ndarray
        """
        raise NotImplementedError

    def on_train_loader_start(self, context: PhaseContext) -> None:
        self._reset()

    def on_train_batch_end(self, context: PhaseContext) -> None:
        if self.enable_on_train_loader and (context.epoch + 1) % self.freq == 0:
            self._on_batch_end(context)

    def on_train_loader_end(self, context: PhaseContext) -> None:
        if self.enable_on_train_loader and (context.epoch + 1) % self.freq == 0:
            self._gather_extreme_batch_images_and_log(context, "train")
            self._reset()

    def on_validation_loader_start(self, context: PhaseContext) -> None:
        self._reset()

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        if self.enable_on_valid_loader and (context.epoch + 1) % self.freq == 0:
            self._on_batch_end(context)

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        if self.enable_on_valid_loader and (context.epoch + 1) % self.freq == 0:
            self._gather_extreme_batch_images_and_log(context, "valid")
            self._reset()

    def _gather_extreme_batch_images_and_log(self, context, loader_name: str):
        input_images_to_save = self.process_extreme_batch()
        images_to_save = maybe_all_gather_as_list(input_images_to_save)
        images_to_save: List[np.ndarray] = list(itertools.chain(*images_to_save))

        if not context.ddp_silent_mode:
            if self.max_images > 0:
                images_to_save = images_to_save[: self.max_images]

            # Before saving images to logger we need to pad them to the same size
            max_height = max([image.shape[0] for image in images_to_save])
            max_width = max([image.shape[1] for image in images_to_save])
            images_to_save = [
                cv2.copyMakeBorder(image, 0, max_height - image.shape[0], 0, max_width - image.shape[1], cv2.BORDER_CONSTANT, value=0)
                for image in images_to_save
            ]
            images_to_save = np.stack(images_to_save, axis=0)

            context.sg_logger.add_images(tag=f"{loader_name}/{self._tag}", images=images_to_save, global_step=context.epoch, data_format="NHWC")

    def _on_batch_end(self, context: PhaseContext) -> None:
        if self.metric is not None:
            self.metric.update(**context.__dict__)
            score = self.metric.compute()
            if self.metric_component_name is not None:
                if not isinstance(score, Mapping) or (isinstance(score, Mapping) and self.metric_component_name not in score.keys()):
                    raise RuntimeError(
                        f"metric_component_name: {self.metric_component_name} is not a component of the monitored metric: {self.metric.__class__.__name__}"
                    )
                score = score[self.metric_component_name]
            elif len(score) > 1:
                raise RuntimeError(f"returned multiple values from {self.metric} but no metric_component_name has been passed to __init__.")
            else:
                score = score.pop(list(score.keys())[0])
            self.metric.reset()

        else:
            # FOR LOSS VALUES, GET THE RIGHT COMPONENT, DERIVE IT ON THE FIRST PASS
            loss_tuple = context.loss_log_items
            if self._first_call:
                self._init_loss_attributes(context)
            score = loss_tuple[self._idx_loss_tuple].detach().cpu().item()

            # IN CONTRARY TO METRICS - LOSS VALUES NEED TO BE REDUCES IN DDP
            device = infer_model_device(context.net)
            score = torch.tensor(score, device=device)
            score = maybe_all_reduce_tensor_average(score)

        if self._is_more_extreme(score):
            self.extreme_score = tensor_container_to_device(score, device="cpu", detach=True, non_blocking=False)
            self.extreme_batch = tensor_container_to_device(context.inputs, device="cpu", detach=True, non_blocking=False)
            self.extreme_preds = tensor_container_to_device(context.preds, device="cpu", detach=True, non_blocking=False)
            self.extreme_targets = tensor_container_to_device(context.target, device="cpu", detach=True, non_blocking=False)
            self.extreme_additional_batch_items = tensor_container_to_device(context.additional_batch_items, device="cpu", detach=True, non_blocking=False)

    def _init_loss_attributes(self, context: PhaseContext):
        if self.loss_to_monitor not in context.loss_logging_items_names:
            raise ValueError(f"{self.loss_to_monitor} not a loss or loss component.")
        self._idx_loss_tuple = context.loss_logging_items_names.index(self.loss_to_monitor)
        self._first_call = False

    def _reset(self):
        self.extreme_score = None
        self.extreme_batch = None
        self.extreme_preds = None
        self.extreme_targets = None
        self.extreme_additional_batch_items = None
        if self.metric is not None:
            self.metric.reset()

    def _is_more_extreme(self, score: Union[float, torch.Tensor]) -> bool:
        """
        Checks whether computed score is the more extreme than the current extreme score.
        If the current score is None (first call), returns True.
        :param score: A newly computed score.
        :return:      True if score is more extreme than the current extreme score, False otherwise.
        """
        # A score can be Nan/Inf (rare but possible event when training diverges).
        # In such case the both < and > operators would return False according to IEEE 754.
        # As a consequence, self.extreme_inputs / self.extreme_outputs would not be updated
        # and that would crash at the attempt to visualize batch.
        if self.extreme_score is None:
            return True

        if self.max:
            return self.extreme_score < score
        else:
            return self.extreme_score > score


@register_callback("ExtremeBatchDetectionVisualizationCallback")
class ExtremeBatchDetectionVisualizationCallback(ExtremeBatchCaseVisualizationCallback):
    """
    ExtremeBatchSegVisualizationCallback

    Visualizes worst/best batch in an epoch for Object detection.
    For clarity, the batch is saved twice in the SG Logger, once with the model's predictions and once with
     ground truth targets.

    Assumptions on bbox dormats:
     - After applying post_prediction_callback on context.preds, the predictions are a list/Tensor s.t:
        predictions[i] is a tensor of shape nx6 - (x1, y1, x2, y2, confidence, class) where x and y are in pixel units.

     - context.targets is a tensor of shape (total_num_targets, 6), in LABEL_CXCYWH format:  (index, label, cx, cy, w, h).



    Example usage in Yaml config:

        training_hyperparams:
          phase_callbacks:
            - ExtremeBatchDetectionVisualizationCallback:
                metric:
                  DetectionMetrics_050:
                    score_thres: 0.1
                    top_k_predictions: 300
                    num_cls: ${num_classes}
                    normalize_targets: True
                    post_prediction_callback:
                      _target_: super_gradients.training.models.detection_models.pp_yolo_e.PPYoloEPostPredictionCallback
                      score_threshold: 0.01
                      nms_top_k: 1000
                      max_predictions: 300
                      nms_threshold: 0.7
                metric_component_name: 'mAP@0.50'
                post_prediction_callback:
                  _target_: super_gradients.training.models.detection_models.pp_yolo_e.PPYoloEPostPredictionCallback
                  score_threshold: 0.25
                  nms_top_k: 1000
                  max_predictions: 300
                  nms_threshold: 0.7
                normalize_targets: True

    :param metric: Metric, will be the metric which is monitored.

    :param metric_component_name: In case metric returns multiple values (as Mapping),
     the value at metric.compute()[metric_component_name] will be the one monitored.

    :param loss_to_monitor: str, loss_to_monitor corresponding to the 'criterion' passed through training_params in Trainer.train(...).
     Monitoring loss follows the same logic as metric_to_watch in Trainer.train(..), when watching the loss and should be:

        if hasattr(criterion, "component_names") and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"<COMPONENT_NAME>.

        If a single item is returned rather then a tuple:
            <LOSS_CLASS.__name__>.

        When there is no such attributes and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"Loss_"<IDX>

    :param max:                    bool, Whether to take the batch corresponding to the max value of the metric/loss or
    the minimum (default=False).

    :param freq:                   int, epoch frequency to perform all of the above (default=1).

    :param classes:                List[str], a list of class names corresponding to the class indices for display.
    When None, will try to fetch this through a "classes" attribute of the valdiation dataset. If such attribute does
    not exist an error will be raised (default=None).

    :param normalize_targets:      bool, whether to scale the target bboxes. If the bboxes returned by the validation data loader
     are in pixel values range, this needs to be set to True (default=False)

    :param enable_on_train_loader: Controls whether to enable this callback on the train loader. Default is False.
    :param enable_on_valid_loader: Controls whether to enable this callback on the valid loader. Default is True.
    :param max_images:             Maximum images to save. If -1, save all images.
    """

    def __init__(
        self,
        post_prediction_callback: DetectionPostPredictionCallback,
        metric: Optional[Metric] = None,
        metric_component_name: Optional[str] = None,
        loss_to_monitor: Optional[str] = None,
        max: bool = False,
        freq: int = 1,
        classes: Optional[List[str]] = None,
        normalize_targets: bool = False,
        enable_on_train_loader: bool = False,
        enable_on_valid_loader: bool = True,
        max_images: int = -1,
    ):
        super(ExtremeBatchDetectionVisualizationCallback, self).__init__(
            metric=metric,
            metric_component_name=metric_component_name,
            loss_to_monitor=loss_to_monitor,
            max=max,
            freq=freq,
            enable_on_valid_loader=enable_on_valid_loader,
            enable_on_train_loader=enable_on_train_loader,
            max_images=max_images,
        )
        self.post_prediction_callback = post_prediction_callback
        if classes is None:
            logger.info(
                "No classes have been passed to ExtremeBatchDetectionVisualizationCallback. "
                "Will try to fetch them through context.valid_loader.dataset classes attribute if it exists."
            )
        self.classes = list(classes) if classes is not None else None
        self.normalize_targets = normalize_targets

    @staticmethod
    def universal_undo_preprocessing_fn(inputs: torch.Tensor) -> np.ndarray:
        """
        A universal reversing of preprocessing to be passed to DetectionVisualization.visualize_batch's undo_preprocessing_func kwarg.
        This function scales input tensor to 0..255 range, and cast it to uint8 dtype.

        :param inputs: Input 4D tensor of images in BCHW format with unknown normalization.
        :return:       Numpy 4D tensor of images in BHWC format, normalized to 0..255 range (uint8).
        """
        inputs -= inputs.min()
        inputs /= inputs.max() + 1e-8
        inputs *= 255
        inputs = inputs.to(torch.uint8)
        inputs = inputs.cpu().numpy()
        inputs = inputs[:, ::-1, :, :].transpose(0, 2, 3, 1)
        inputs = np.ascontiguousarray(inputs, dtype=np.uint8)
        return inputs

    def process_extreme_batch(self) -> np.ndarray:
        """
        Processes the extreme batch, and returns list of images for visualization.
        Default implementations stacks GT and prediction overlays horisontally.

        :return: np.ndarray A 4D tensor of BHWC shape with visualizations of the extreme batch.
        """
        inputs = self.extreme_batch
        preds = self.post_prediction_callback(self.extreme_preds, self.extreme_batch.device)
        targets = self.extreme_targets.clone()
        if self.normalize_targets:
            target_bboxes = targets[:, 2:]
            target_bboxes = cxcywh2xyxy(target_bboxes)
            _, _, height, width = inputs.shape
            target_bboxes[:, [0, 2]] /= width
            target_bboxes[:, [1, 3]] /= height
            target_bboxes = xyxy2cxcywh(target_bboxes)
            targets[:, 2:] = target_bboxes

        images_to_save_preds = DetectionVisualization.visualize_batch(
            inputs, preds, targets, "extreme_batch_preds", self.classes, gt_alpha=0.0, undo_preprocessing_func=self.universal_undo_preprocessing_fn
        )
        images_to_save_preds = np.stack(images_to_save_preds)

        images_to_save_gt = DetectionVisualization.visualize_batch(
            inputs, None, targets, "extreme_batch_gt", self.classes, gt_alpha=1.0, undo_preprocessing_func=self.universal_undo_preprocessing_fn
        )
        images_to_save_gt = np.stack(images_to_save_gt)

        # Stack the predictions and GT images together
        return np.concatenate([images_to_save_gt, images_to_save_preds], axis=2)

    def on_validation_loader_start(self, context: PhaseContext) -> None:
        if self.classes is None:
            if hasattr(context.valid_loader.dataset, "classes"):
                self.classes = context.valid_loader.dataset.classes
            else:
                raise RuntimeError("Couldn't fetch classes from valid_loader, please pass classes explicitly")
        super().on_validation_loader_start(context)


@register_callback("ExtremeBatchSegVisualizationCallback")
class ExtremeBatchSegVisualizationCallback(ExtremeBatchCaseVisualizationCallback):
    """
    ExtremeBatchSegVisualizationCallback

    Visualizes worst/best batch in an epoch, for segmentation.
    Assumes context.preds in validation is a score tensor of shape BCHW, or a tuple whose first item is one.

    True predictions will be marked with green, false ones with red.

    Example usage in training_params definition:

        training_hyperparams ={
          ...
          "phase_callbacks":
            [ExtremeBatchSegVisualizationCallback(
                metric=IoU(20, ignore_idx=19)
                max=False
                ignore_idx=19),
            ExtremeBatchSegVisualizationCallback(
                loss_to_monitor="CrossEntropyLoss"
                max=True
                ignore_idx=19)]
                ...}

    Example usage in Yaml config:

        training_hyperparams:
          phase_callbacks:
            - ExtremeBatchSegVisualizationCallback:
                loss_to_monitor: DiceCEEdgeLoss/aux_loss0
                ignore_idx: 19

    :param metric: Metric, will be the metric which is monitored.

    :param metric_component_name: In case metric returns multiple values (as Mapping),
     the value at metric.compute()[metric_component_name] will be the one monitored.

    :param loss_to_monitor: str, loss_to_monitor corresponding to the 'criterion' passed through training_params in Trainer.train(...).
     Monitoring loss follows the same logic as metric_to_watch in Trainer.train(..), when watching the loss and should be:

        if hasattr(criterion, "component_names") and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"<COMPONENT_NAME>.

        If a single item is returned rather then a tuple:
            <LOSS_CLASS.__name__>.

        When there is no such attributes and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"Loss_"<IDX>

    :param max:                    bool, Whether to take the batch corresponding to the max value of the metric/loss or
    the minimum (default=False).

    :param freq:                   int, epoch frequency to perform all of the above (default=1).

    :param enable_on_train_loader: Controls whether to enable this callback on the train loader. Default is False.
    :param enable_on_valid_loader: Controls whether to enable this callback on the valid loader. Default is True.
    :param max_images:             Maximum images to save. If -1, save all images.
    """

    def __init__(
        self,
        metric: Optional[Metric] = None,
        metric_component_name: Optional[str] = None,
        loss_to_monitor: Optional[str] = None,
        max: bool = False,
        freq: int = 1,
        ignore_idx: int = -1,
        enable_on_train_loader: bool = False,
        enable_on_valid_loader: bool = True,
        max_images: int = -1,
    ):
        super(ExtremeBatchSegVisualizationCallback, self).__init__(
            metric=metric,
            metric_component_name=metric_component_name,
            loss_to_monitor=loss_to_monitor,
            max=max,
            freq=freq,
            enable_on_valid_loader=enable_on_valid_loader,
            enable_on_train_loader=enable_on_train_loader,
            max_images=max_images,
        )
        self.ignore_idx = ignore_idx

    @torch.no_grad()
    def process_extreme_batch(self) -> np.ndarray:
        inputs = self.extreme_batch
        inputs -= inputs.min()
        inputs /= inputs.max()
        inputs *= 255
        inputs = inputs.to(torch.uint8)
        preds = self.extreme_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.argmax(1)
        p_mask = preds == self.extreme_targets
        n_mask = preds != self.extreme_targets
        p_mask[self.extreme_targets == self.ignore_idx] = False
        n_mask[self.extreme_targets == self.ignore_idx] = False
        overlay = torch.cat([p_mask.unsqueeze(1), n_mask.unsqueeze(1)], 1)
        colors = ["green", "red"]
        images_to_save = []
        for i in range(len(inputs)):
            image = draw_segmentation_masks(inputs[i].cpu(), overlay[i].cpu(), colors=colors, alpha=0.4).numpy()
            image = np.transpose(image, (1, 2, 0))
            images_to_save.append(image)
        images_to_save = np.stack(images_to_save)
        return images_to_save
