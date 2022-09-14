from super_gradients.training.utils.callbacks.callbacks import DeciLabUploadCallback, LRCallbackBase, LRSchedulerCallback, MetricsUpdateCallback, \
    ModelConversionCheckCallback, YoloXTrainingStageSwitchCallback, StepLRCallback, PolyLRCallback,\
    CosineLRCallback, ExponentialLRCallback, FunctionLRCallback, WarmupLRCallback

from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.datasets.datasets_utils import DetectionMultiscalePrePredictionCallback


class CallbackNames:
    """Static class holding all the supported callback names"""""
    DECI_LAB_UPLOAD = 'DeciLabUploadCallback'
    LR_CALLBACK_BASE = 'LRCallbackBase'
    LR_SCHEDULER = 'LRSchedulerCallback'
    METRICS_UPDATE = 'MetricsUpdateCallback'
    MODEL_CONVERSION_CHECK = 'ModelConversionCheckCallback'
    EARLY_STOP = 'EarlyStop'
    DETECTION_MULTISCALE_PREPREDICTION = 'DetectionMultiscalePrePredictionCallback'
    YOLOX_TRAINING_STAGE_SWITCH = 'YoloXTrainingStageSwitchCallback'


CALLBACKS = {
    CallbackNames.DECI_LAB_UPLOAD: DeciLabUploadCallback,
    CallbackNames.LR_CALLBACK_BASE: LRCallbackBase,
    CallbackNames.LR_SCHEDULER: LRSchedulerCallback,
    CallbackNames.METRICS_UPDATE: MetricsUpdateCallback,
    CallbackNames.MODEL_CONVERSION_CHECK: ModelConversionCheckCallback,
    CallbackNames.EARLY_STOP: EarlyStop,
    CallbackNames.DETECTION_MULTISCALE_PREPREDICTION: DetectionMultiscalePrePredictionCallback,
    CallbackNames.YOLOX_TRAINING_STAGE_SWITCH: YoloXTrainingStageSwitchCallback
}


# DICT FOR LEGACY LR HARD-CODED REGIMES, WILL BE DELETED IN THE FUTURE
class LRScheduler:
    """Static class to hold all the supported LR Scheduler names"""""
    STEP = "step"
    POLY = "poly"
    COSINE = "cosine"
    EXP = "exp"
    FUNCTION = "function"


LR_SCHEDULERS_CLS_DICT = {
    LRScheduler.STEP: StepLRCallback,
    LRScheduler.POLY: PolyLRCallback,
    LRScheduler.COSINE: CosineLRCallback,
    LRScheduler.EXP: ExponentialLRCallback,
    LRScheduler.FUNCTION: FunctionLRCallback,
}


# DICT FOR LEGACY LR HARD-CODED REGIMES, WILL BE DELETED IN THE FUTURE
class LRWarmup:
    """Static class to hold all the supported LR Warmup names"""""
    LINEAR_STEP = "linear_step"


LR_WARMUP_CLS_DICT = {LRWarmup.LINEAR_STEP: WarmupLRCallback}
