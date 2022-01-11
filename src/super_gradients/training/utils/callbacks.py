import copy
import getpass
import math
import os
from enum import Enum
import math
from super_gradients.training.utils.utils import get_filename_suffix_by_framework, get_param
import torch
import numpy as np
import onnxruntime
import onnx
import onnxruntime
import torch

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.utils import get_filename_suffix_by_framework

logger = get_logger(__name__)

try:
    from deci_lab_client.client import DeciPlatformClient
    from deci_lab_client.models import ModelBenchmarkState

    _imported_deci_lab_failiure = None
except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.warn('Failed to import deci_lab_client')
    _imported_deci_lab_failiure = import_err


class Phase(Enum):
    PRE_TRAINING = "PRE_TRAINING"
    TRAIN_BATCH_END = "TRAIN_BATCH_END"
    TRAIN_BATCH_STEP = "TRAIN_BATCH_STEP"
    TRAIN_EPOCH_START = "TRAIN_EPOCH_START"
    TRAIN_EPOCH_END = "TRAIN_EPOCH_END"
    VALIDATION_BATCH_END = "VALIDATION_BATCH_END"
    VALIDATION_EPOCH_END = "VALIDATION_EPOCH_END"
    VALIDATION_END_BEST_EPOCH = "VALIDATION_END_BEST_EPOCH"
    TEST_BATCH_END = "TEST_BATCH_END"
    TEST_END = "TEST_END"
    POST_TRAINING = "POST_TRAINING"


class PhaseContext:
    """
    Represents the input for phase callbacks, and is constantly updated after callback calls.

    """

    def __init__(self, epoch=None, batch_idx=None, optimizer=None, metrics_dict=None, inputs=None, preds=None,
                 target=None, metrics_compute_fn=None, loss_avg_meter=None, loss_log_items=None, criterion=None,
                 device=None, experiment_name=None, ckpt_dir=None, net=None, lr_warmup_epochs=None, sg_logger=None):
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.optimizer = optimizer
        self.inputs = inputs
        self.preds = preds
        self.target = target
        self.metrics_dict = metrics_dict
        self.metrics_compute_fn = metrics_compute_fn
        self.loss_avg_meter = loss_avg_meter
        self.loss_log_items = loss_log_items
        self.criterion = criterion
        self.device = device
        self.stop_training = False
        self.experiment_name = experiment_name
        self.ckpt_dir = ckpt_dir
        self.net = net
        self.lr_warmup_epochs = lr_warmup_epochs
        self.sg_logger = sg_logger

    def update_context(self, **kwargs):
        for attr, attr_val in kwargs.items():
            setattr(self, attr, attr_val)


class PhaseCallback:
    def __init__(self, phase: Phase):
        self.phase = phase

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class ModelConversionCheckCallback(PhaseCallback):
    """
    Pre-training callback that verifies model conversion to onnx given specified conversion parameters.

    The model is converted, then inference is applied with onnx runtime.

    Use this callback wit hthe same args as DeciPlatformCallback to prevent conversion fails at the end of training.

    Attributes:

        model_meta_data: (ModelMetadata) model's meta-data object.

        The following parameters may be passed as kwargs in order to control the conversion to onnx:
        :param opset_version (default=11)
        :param do_constant_folding (default=True)
        :param dynamic_axes (default=
                                        {'input': {0: 'batch_size'},
                                        # Variable length axes
                                        'output': {0: 'batch_size'}}

                                        )
        :param input_names (default=["input"])
        :param output_names (default=["output"])
    """

    def __init__(self, model_meta_data, **kwargs):
        super(ModelConversionCheckCallback, self).__init__(phase=Phase.PRE_TRAINING)
        self.model_meta_data = model_meta_data

        self.opset_version = kwargs.get('opset_version') or 10
        self.do_constant_folding = kwargs.get('do_constant_folding', None) if kwargs.get('do_constant_folding',
                                                                                         None) else True
        self.input_names = kwargs.get('input_names') or ['input']
        self.output_names = kwargs.get('output_names') or ['output']
        self.dynamic_axes = kwargs.get('dynamic_axes') or {'input': {0: 'batch_size'},
                                                           'output': {0: 'batch_size'}}

    def __call__(self, context: PhaseContext):
        model = copy.deepcopy(context.net.module)
        model = model.cpu()
        x = torch.randn(self.model_meta_data.primary_batch_size, *self.model_meta_data.input_dimensions,
                        requires_grad=False)

        tmp_model_path = os.path.join(context.ckpt_dir, self.model_meta_data.name + '_tmp.onnx')
        model.eval()  # Put model into eval mode

        with torch.no_grad():
            torch_out = model(x)

        torch.onnx.export(model,  # Model being run
                          x,  # Model input (or a tuple for multiple inputs)
                          tmp_model_path,  # Where to save the model (can be a file or file-like object)
                          export_params=True,  # Store the trained parameter weights inside the model file
                          opset_version=self.opset_version,
                          do_constant_folding=self.do_constant_folding,
                          input_names=self.input_names,
                          output_names=self.output_names,
                          dynamic_axes=self.dynamic_axes)

        onnx_model = onnx.load(tmp_model_path)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(tmp_model_path,
                                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(torch_out.cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

        os.remove(tmp_model_path)

        logger.info("Exported model has been tested with ONNXRuntime, and the result looks good!")


class DeciLabUploadCallback(PhaseCallback):
    """
    Post-training callback for uploading and optimizing a model.

    Attributes:

        email: (str) username for Deci platform.
        model_meta_data: (ModelMetadata) model's meta-data object.
        optimization_request_form: (dict) optimization request form object.
        password: (str) default=None, should only be used for testing.
        ckpt_name: (str) default="ckpt_best" refers to the filename of the checkpoint, inside the checkpoint directory.

        The following parameters may be passed as kwargs in order to control the conversion to onnx:
        :param opset_version
        :param do_constant_folding
        :param dynamic_axes
        :param input_names
        :param output_names
    """

    def __init__(self, email, model_meta_data, optimization_request_form, password=None, ckpt_name="ckpt_best.pth",
                 **kwargs):
        super().__init__(phase=Phase.POST_TRAINING)
        if _imported_deci_lab_failiure is not None:
            raise _imported_deci_lab_failiure
        self.model_meta_data = model_meta_data
        self.optimization_request_form = optimization_request_form
        self.conversion_kwargs = kwargs
        self.ckpt_name = ckpt_name
        self.platform_client = DeciPlatformClient('api.deci.ai', 443, https=True)

        password = password or getpass.getpass()
        self.platform_client.login(email, password)

    def __call__(self, context: PhaseContext):
        try:
            model = copy.deepcopy(context.net)
            model_state_dict_path = os.path.join(context.ckpt_dir, self.ckpt_name)['net']
            model.load_state_dict(model_state_dict_path)

            self.platform_client.add_model(self.model_meta_data,
                                           local_loaded_model=model.module.cpu(),
                                           optimization_request=self.optimization_request_form,
                                           **self.conversion_kwargs)

            new_model_from_repo_name = self.model_meta_data.name + '_1_1'
            finished = False
            while not finished:
                your_model_from_repo = self.platform_client.get_model_by_name(name=new_model_from_repo_name).data
                if your_model_from_repo.benchmark_state not in [ModelBenchmarkState.IN_PROGRESS,
                                                                ModelBenchmarkState.PENDING]:
                    finished = True
            logger.info('successfully added ' + str(your_model_from_repo.name) + ' to model repository')

            filename_ext = get_filename_suffix_by_framework(self.model_meta_data.framework)
            download_path = os.path.join(context.ckpt_dir, new_model_from_repo_name + '_optimized' + filename_ext)
            self.platform_client.download_model(your_model_from_repo.model_id, download_to_path=download_path)
        except Exception as ex:
            logger.error(ex)


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

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def update_lr(self, optimizer, epoch, batch_idx=None):
        if self.update_param_groups:
            param_groups = self.net.module.update_param_groups(optimizer.param_groups, self.lr, epoch, batch_idx,
                                                               self.training_params, self.train_loader_len)
            optimizer.param_groups = param_groups
        else:
            # UPDATE THE OPTIMIZERS PARAMETER
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr


class WarmupLRCallback(LRCallbackBase):
    """
    LR scheduling callback for linear step warmup.
    LR climbs from warmup_initial_lr with even steps to initial lr. When warmup_initial_lr is None- LR climb starts from
     initial_lr/(1+warmup_epochs).

    """

    def __init__(self, **kwargs):
        super(WarmupLRCallback, self).__init__(Phase.TRAIN_EPOCH_START, **kwargs)
        self.warmup_initial_lr = self.training_params.warmup_initial_lr or self.initial_lr / (self.training_params.lr_warmup_epochs + 1)
        self.warmup_step_size = (self.initial_lr - self.warmup_initial_lr) / self.training_params.lr_warmup_epochs

    def __call__(self, context: PhaseContext):
        if self.training_params.lr_warmup_epochs >= context.epoch:
            self.lr = self.warmup_initial_lr + context.epoch * self.warmup_step_size
            self.update_lr(context.optimizer, context.epoch, None)


class YoloV5WarmupLRCallback(LRCallbackBase):
    def __init__(self, **kwargs):
        super(YoloV5WarmupLRCallback, self).__init__(Phase.TRAIN_BATCH_END, **kwargs)

    def __call__(self, context, **kwargs):
        lr_warmup_epochs = get_param(self.training_params, 'lr_warmup_epochs', 0)
        if context.epoch < self.training_params.lr_warmup_epochs:
            # OVERRIDE THE lr FROM DeciModelBase WITH initial_lr, SINCE DeciModelBase MANIPULATE THE ORIGINAL VALUE
            lr = self.training_params.initial_lr
            momentum = get_param(self.training_params.optimizer_params, 'momentum')
            warmup_momentum = get_param(self.training_params, 'warmup_momentum', momentum)
            warmup_bias_lr = get_param(self.training_params, 'warmup_bias_lr', lr)
            nw = lr_warmup_epochs * self.train_loader_len
            ni = context.epoch * self.train_loader_len + context.batch_idx
            xi = [0, nw]  # x interp
            for x in context.optimizer.param_groups:
                # BIAS LR FALLS FROM 0.1 TO LR0, ALL OTHER LRS RISE FROM 0.0 TO LR0
                x['lr'] = np.interp(ni, xi, [warmup_bias_lr if x['name'] == 'bias' else 0.0, lr])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])


class StepLRCallback(LRCallbackBase):
    """
    Hard coded step learning rate scheduling (i.e at specific milestones).
    """

    def __init__(self, lr_updates, lr_decay_factor, step_lr_update_freq=None, **kwargs):
        super(StepLRCallback, self).__init__(Phase.TRAIN_EPOCH_END, **kwargs)
        if step_lr_update_freq and len(lr_updates):
            raise ValueError("Only one of [lr_updates, step_lr_update_freq] should be passed to StepLRCallback constructor")

        if step_lr_update_freq:
            max_epochs = self.training_params.max_epochs
            warmup_epochs = self.training_params.lr_warmup_epochs
            lr_updates = [int(np.ceil(step_lr_update_freq * x)) for x in range(1, max_epochs) if warmup_epochs <= int(np.ceil(step_lr_update_freq * x)) < max_epochs]
        self.lr_updates = lr_updates
        self.lr_decay_factor = lr_decay_factor

    def __call__(self, context: PhaseContext):
        if self.training_params.lr_warmup_epochs <= context.epoch:
            num_updates_passed = [x for x in self.lr_updates if x <= context.epoch]
            self.lr = self.initial_lr * self.lr_decay_factor ** len(num_updates_passed)
            self.update_lr(context.optimizer, context.epoch, None)


class PolyLRCallback(LRCallbackBase):
    """
    Hard coded polynomial decay learning rate scheduling (i.e at specific milestones).
    """

    def __init__(self, max_epochs, **kwargs):
        super(PolyLRCallback, self).__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        self.max_epochs = max_epochs

    def __call__(self, context: PhaseContext):
        # POLYNOMIAL LEARNING RATE
        if self.training_params.lr_warmup_epochs <= context.epoch:
            effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
            effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs

            current_iter = (self.train_loader_len * effective_epoch + context.batch_idx) / self.training_params.batch_accumulate
            max_iter = self.train_loader_len * effective_max_epochs / self.training_params.batch_accumulate
            self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)), 0.9)
            self.update_lr(context.optimizer, context.epoch, context.batch_idx)


class CosineLRCallback(LRCallbackBase):
    """
    Hard coded step Cosine anealing learning rate scheduling.
    """

    def __init__(self, max_epochs, cosine_final_lr_ratio, **kwargs):
        super(CosineLRCallback, self).__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        self.max_epochs = max_epochs
        self.cosine_final_lr_ratio = cosine_final_lr_ratio

    def __call__(self, context: PhaseContext):
        # COSINE LEARNING RATE
        if self.training_params.lr_warmup_epochs <= context.epoch:
            effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
            effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs

            current_iter = self.train_loader_len * effective_epoch + context.batch_idx
            max_iter = self.train_loader_len * effective_max_epochs
            lr = 0.5 * self.initial_lr * (1.0 + math.cos(current_iter / (max_iter + 1) * math.pi))
            # the cosine starts from initial_lr and reaches initial_lr * cosine_final_lr_ratio in last epoch
            self.lr = lr * (1 - self.cosine_final_lr_ratio) + (self.initial_lr * self.cosine_final_lr_ratio)
            self.update_lr(context.optimizer, context.epoch, context.batch_idx)


class FunctionLRCallback(LRCallbackBase):
    """
    Hard coded rate scheduling for user defined lr scheduling function.
    """

    def __init__(self, max_epochs, lr_schedule_function, **kwargs):
        super(FunctionLRCallback, self).__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        assert callable(self.lr_schedule_function), 'self.lr_function must be callable'
        self.lr_schedule_function = lr_schedule_function
        self.max_epochs = max_epochs

    def __call__(self, context: PhaseContext):
        if self.training_params.lr_warmup_epochs <= context.epoch:
            effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
            effective_max_epochs = self.max_epochs - self.training_params.lr_warmup_epochs

            self.lr = self.lr_schedule_function(initial_lr=self.initial_lr, epoch=effective_epoch,
                                                iter=context.batch_idx,
                                                max_epoch=effective_max_epochs,
                                                iters_per_epoch=self.train_loader_len)
            self.update_lr(context.optimizer, context.epoch, context.batch_idx)


class IllegalLRSchedulerMetric(Exception):
    """Exception raised illegal combination of training parameters.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, metric_name, metrics_dict):
        self.message = "Illegal metric name: " + metric_name + ". Expected one of metics_dics keys: " + str(
            metrics_dict.keys())
        super().__init__(self.message)


class LRSchedulerCallback(PhaseCallback):
    """
    Learning rate scheduler callback.

    Attributes:
        scheduler: torch.optim._LRScheduler, the learning rate scheduler to be called step() with.
        metric_name: str, (default=None) the metric name for ReduceLROnPlateau learning rate scheduler.

        When passing __call__ a metrics_dict, with a key=self.metric_name, the value of that metric will monitored
         for ReduceLROnPlateau (i.e step(metrics_dict[self.metric_name]).
    """

    def __init__(self, scheduler, phase, metric_name=None):
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


class MetricsUpdateCallback(PhaseCallback):
    def __init__(self, phase: Phase):
        super(MetricsUpdateCallback, self).__init__(phase)

    def __call__(self, context: PhaseContext):
        context.metrics_compute_fn.update(**context.__dict__)
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


class CallbackHandler:
    """
    Runs all callbacks who's phase attribute equals to the given phase.

    Attributes:
        callbacks: List[PhaseCallback]. Callbacks to be run.
    """

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, phase: Phase, context: PhaseContext):
        for callback in self.callbacks:
            if callback.phase == phase:
                callback(context)


# DICT FOR LEGACY LR HARD-CODED REGIMES, WILL BE DELETED IN THE FUTURE
LR_SCHEDULERS_CLS_DICT = {"step": StepLRCallback,
                          "poly": PolyLRCallback,
                          "cosine": CosineLRCallback,
                          "function": FunctionLRCallback
                          }

LR_WARMUP_CLS_DICT = {"linear_step": WarmupLRCallback,
                      "yolov5_warmup": YoloV5WarmupLRCallback}
