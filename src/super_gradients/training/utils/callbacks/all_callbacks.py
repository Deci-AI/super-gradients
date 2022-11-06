from super_gradients.common.object_names import Callbacks, LRSchedulers, LRWarmups
from super_gradients.training.utils.callbacks.callbacks import (
    DeciLabUploadCallback,
    LRCallbackBase,
    LRSchedulerCallback,
    MetricsUpdateCallback,
    ModelConversionCheckCallback,
    YoloXTrainingStageSwitchCallback,
    StepLRCallback,
    PolyLRCallback,
    CosineLRCallback,
    ExponentialLRCallback,
    FunctionLRCallback,
    WarmupLRCallback,
)

from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.datasets.datasets_utils import DetectionMultiscalePrePredictionCallback


CALLBACKS = {
    Callbacks.DECI_LAB_UPLOAD: DeciLabUploadCallback,
    Callbacks.LR_CALLBACK_BASE: LRCallbackBase,
    Callbacks.LR_SCHEDULER: LRSchedulerCallback,
    Callbacks.METRICS_UPDATE: MetricsUpdateCallback,
    Callbacks.MODEL_CONVERSION_CHECK: ModelConversionCheckCallback,
    Callbacks.EARLY_STOP: EarlyStop,
    Callbacks.DETECTION_MULTISCALE_PREPREDICTION: DetectionMultiscalePrePredictionCallback,
    Callbacks.YOLOX_TRAINING_STAGE_SWITCH: YoloXTrainingStageSwitchCallback,
}


LR_SCHEDULERS_CLS_DICT = {
    LRSchedulers.STEP: StepLRCallback,
    LRSchedulers.POLY: PolyLRCallback,
    LRSchedulers.COSINE: CosineLRCallback,
    LRSchedulers.EXP: ExponentialLRCallback,
    LRSchedulers.FUNCTION: FunctionLRCallback,
}


LR_WARMUP_CLS_DICT = {LRWarmups.LINEAR_STEP: WarmupLRCallback}
