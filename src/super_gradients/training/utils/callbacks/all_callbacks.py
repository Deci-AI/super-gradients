from super_gradients.common.object_names import Callbacks, LRSchedulers, LRWarmups
from super_gradients.training.datasets.datasets_utils import DetectionMultiscalePrePredictionCallback
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
    EpochStepWarmupLRCallback,
    BatchStepLinearWarmupLRCallback,
)
from super_gradients.training.utils.deprecated_utils import wrap_with_warning
from super_gradients.training.utils.early_stopping import EarlyStop

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


LR_WARMUP_CLS_DICT = {
    LRWarmups.LINEAR_STEP: wrap_with_warning(
        EpochStepWarmupLRCallback,
        message=f"Parameter {LRWarmups.LINEAR_STEP} has been made deprecated and will be removed in the next SG release. "
        f"Please use `{LRWarmups.LINEAR_EPOCH_STEP}` instead.",
    ),
    LRWarmups.LINEAR_EPOCH_STEP: EpochStepWarmupLRCallback,
    LRWarmups.LINEAR_BATCH_STEP: BatchStepLinearWarmupLRCallback,
}
