import copy
import math
import os
import signal
import time
from typing import List, Union, Optional, Sequence

import csv
import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from deprecated import deprecated

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.plugins.deci_client import DeciClient
from super_gradients.common.registry.registry import register_lr_scheduler, register_lr_warmup, register_callback
from super_gradients.common.object_names import LRSchedulers, LRWarmups, Callbacks
from super_gradients.common.sg_loggers.time_units import GlobalBatchStepNumber, EpochNumber
from super_gradients.training.utils.callbacks.base_callbacks import PhaseCallback, PhaseContext, Phase, Callback
from super_gradients.training.utils.detection_utils import DetectionVisualization, DetectionPostPredictionCallback
from super_gradients.training.utils.segmentation_utils import BinarySegmentationVisualization
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.common.environment.checkpoints_dir_utils import get_project_checkpoints_dir_path


logger = get_logger(__name__)


class ContextSgMethods:
    """
    Class for delegating Trainer's methods, so that only the relevant ones are ("phase wise") are accessible.
    """

    def __init__(self, **methods):
        for attr, attr_val in methods.items():
            setattr(self, attr, attr_val)


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
        model = copy.deepcopy(context.net.module)
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
            model = copy.deepcopy(context.net)
            model_state_dict_path = os.path.join(context.ckpt_dir, self.ckpt_name)
            model_state_dict = torch.load(model_state_dict_path)["net"]
            model.load_state_dict(state_dict=model_state_dict)

            model = model.module.cpu()
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
        self.initial_lr = initial_lr
        self.lr = initial_lr
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
        if self.update_param_groups:
            param_groups = self.net.module.update_param_groups(optimizer.param_groups, self.lr, epoch, batch_idx, self.training_params, self.train_loader_len)
            optimizer.param_groups = param_groups
        else:
            # UPDATE THE OPTIMIZERS PARAMETER
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr


@register_lr_warmup(LRWarmups.LINEAR_EPOCH_STEP)
class EpochStepWarmupLRCallback(LRCallbackBase):
    """
    LR scheduling callback for linear step warmup. This scheduler uses a whole epoch as single step.
    LR climbs from warmup_initial_lr with even steps to initial lr. When warmup_initial_lr is None - LR climb starts from
     initial_lr/(1+warmup_epochs).

    """

    def __init__(self, **kwargs):
        super(EpochStepWarmupLRCallback, self).__init__(Phase.TRAIN_EPOCH_START, **kwargs)
        self.warmup_initial_lr = self.training_params.warmup_initial_lr or self.initial_lr / (self.training_params.lr_warmup_epochs + 1)
        self.warmup_step_size = (
            (self.initial_lr - self.warmup_initial_lr) / self.training_params.lr_warmup_epochs if self.training_params.lr_warmup_epochs > 0 else 0
        )

    def perform_scheduling(self, context):
        self.lr = self.warmup_initial_lr + context.epoch * self.warmup_step_size
        self.update_lr(context.optimizer, context.epoch, None)

    def is_lr_scheduling_enabled(self, context):
        return self.training_params.lr_warmup_epochs > 0 and self.training_params.lr_warmup_epochs >= context.epoch


@register_lr_warmup(LRWarmups.LINEAR_STEP)
class LinearStepWarmupLRCallback(EpochStepWarmupLRCallback):
    """Deprecated, use EpochStepWarmupLRCallback instead"""

    def __init__(self, **kwargs):
        logger.warning(
            f"Parameter {LRWarmups.LINEAR_STEP} has been made deprecated and will be removed in the next SG release. "
            f"Please use `{LRWarmups.LINEAR_EPOCH_STEP}` instead."
        )
        super(LinearStepWarmupLRCallback, self).__init__(**kwargs)


@register_lr_warmup(LRWarmups.LINEAR_BATCH_STEP)
class BatchStepLinearWarmupLRCallback(Callback):
    """
    LR scheduling callback for linear step warmup on each batch step.
    LR climbs from warmup_initial_lr with to initial lr.
    """

    def __init__(
        self,
        warmup_initial_lr: float,
        initial_lr: float,
        train_loader_len: int,
        update_param_groups: bool,
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

        super(BatchStepLinearWarmupLRCallback, self).__init__()

        if lr_warmup_steps > train_loader_len:
            logger.warning(
                f"Number of warmup steps ({lr_warmup_steps}) is greater than number of steps in epoch ({train_loader_len}). "
                f"Warmup steps will be capped to number of steps in epoch to avoid interfering with any pre-epoch LR schedulers."
            )

        lr_warmup_steps = min(lr_warmup_steps, train_loader_len)
        learning_rates = np.linspace(start=warmup_initial_lr, stop=initial_lr, num=lr_warmup_steps, endpoint=True)

        self.lr = initial_lr
        self.initial_lr = initial_lr
        self.update_param_groups = update_param_groups
        self.training_params = training_params
        self.net = net
        self.learning_rates = learning_rates
        self.train_loader_len = train_loader_len
        self.lr_warmup_steps = lr_warmup_steps

    def on_train_batch_start(self, context: PhaseContext) -> None:
        global_training_step = context.batch_idx + context.epoch * self.train_loader_len
        if global_training_step < self.lr_warmup_steps:
            self.lr = float(self.learning_rates[global_training_step])
            self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def update_lr(self, optimizer, epoch, batch_idx=None):
        """
        Same as in LRCallbackBase
        :param optimizer:
        :param epoch:
        :param batch_idx:
        :return:
        """
        if self.update_param_groups:
            param_groups = self.net.module.update_param_groups(optimizer.param_groups, self.lr, epoch, batch_idx, self.training_params, self.train_loader_len)
            optimizer.param_groups = param_groups
        else:
            # UPDATE THE OPTIMIZERS PARAMETER
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr


@register_lr_scheduler(LRSchedulers.STEP)
class StepLRCallback(LRCallbackBase):
    """
    Hard coded step learning rate scheduling (i.e at specific milestones).
    """

    def __init__(self, lr_updates, lr_decay_factor, step_lr_update_freq=None, **kwargs):
        super(StepLRCallback, self).__init__(Phase.TRAIN_EPOCH_END, **kwargs)
        if step_lr_update_freq and len(lr_updates):
            raise ValueError("Only one of [lr_updates, step_lr_update_freq] should be passed to StepLRCallback constructor")

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
        self.lr = self.initial_lr * self.lr_decay_factor ** len(num_updates_passed)
        self.update_lr(context.optimizer, context.epoch, None)

    def is_lr_scheduling_enabled(self, context):
        return self.training_params.lr_warmup_epochs <= context.epoch


@register_lr_scheduler(LRSchedulers.EXP)
class ExponentialLRCallback(LRCallbackBase):
    """
    Exponential decay learning rate scheduling. Decays the learning rate by `lr_decay_factor` every epoch.
    """

    def __init__(self, lr_decay_factor: float, **kwargs):
        super().__init__(phase=Phase.TRAIN_BATCH_STEP, **kwargs)
        self.lr_decay_factor = lr_decay_factor

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        current_iter = self.train_loader_len * effective_epoch + context.batch_idx
        self.lr = self.initial_lr * self.lr_decay_factor ** (current_iter / self.train_loader_len)
        self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def is_lr_scheduling_enabled(self, context):
        post_warmup_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        return self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs


@register_lr_scheduler(LRSchedulers.POLY)
class PolyLRCallback(LRCallbackBase):
    """
    Hard coded polynomial decay learning rate scheduling (i.e at specific milestones).
    """

    def __init__(self, max_epochs, **kwargs):
        super(PolyLRCallback, self).__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        self.max_epochs = max_epochs

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs - self.training_params.lr_cooldown_epochs
        current_iter = (self.train_loader_len * effective_epoch + context.batch_idx) / self.training_params.batch_accumulate
        max_iter = self.train_loader_len * effective_max_epochs / self.training_params.batch_accumulate
        self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)), 0.9)
        self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def is_lr_scheduling_enabled(self, context):
        post_warmup_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        return self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs


@register_lr_scheduler(LRSchedulers.COSINE)
class CosineLRCallback(LRCallbackBase):
    """
    Hard coded step Cosine anealing learning rate scheduling.
    """

    def __init__(self, max_epochs, cosine_final_lr_ratio, **kwargs):
        super(CosineLRCallback, self).__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        self.max_epochs = max_epochs
        self.cosine_final_lr_ratio = cosine_final_lr_ratio

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs - self.training_params.lr_cooldown_epochs
        current_iter = max(0, self.train_loader_len * effective_epoch + context.batch_idx - self.training_params.lr_warmup_steps)
        max_iter = self.train_loader_len * effective_max_epochs - self.training_params.lr_warmup_steps

        lr = self.compute_learning_rate(current_iter, max_iter, self.initial_lr, self.cosine_final_lr_ratio)
        self.lr = float(lr)
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


@register_lr_scheduler(LRSchedulers.FUNCTION)
class FunctionLRCallback(LRCallbackBase):
    """
    Hard coded rate scheduling for user defined lr scheduling function.
    """

    @deprecated(version="3.2.0", reason="This callback is deprecated and will be removed in future versions.")
    def __init__(self, max_epochs, lr_schedule_function, **kwargs):
        super(FunctionLRCallback, self).__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        assert callable(lr_schedule_function), "self.lr_function must be callable"
        self.lr_schedule_function = lr_schedule_function
        self.max_epochs = max_epochs

    def is_lr_scheduling_enabled(self, context):
        post_warmup_epochs = self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        return self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs - self.training_params.lr_cooldown_epochs
        self.lr = self.lr_schedule_function(
            initial_lr=self.initial_lr,
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

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler, phase: Phase, metric_name: str = None):
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
    def __init__(self, phase: Phase):
        super(MetricsUpdateCallback, self).__init__(phase)

    def __call__(self, context: PhaseContext):
        context.metrics_compute_fn.update(**context.__dict__)
        if context.criterion is not None:
            context.loss_avg_meter.update(context.loss_log_items, len(context.inputs))


class KDModelMetricsUpdateCallback(MetricsUpdateCallback):
    def __init__(self, phase: Phase):
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

    def __init__(self, phase: Phase):
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
        phase: Phase,
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
        if context.epoch % self.freq == 0 and context.batch_idx == self.batch_idx:
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

    def __init__(self, phase: Phase, freq: int, batch_idx: int = 0, last_img_idx_in_batch: int = -1):
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
