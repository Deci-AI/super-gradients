from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.datasets_utils import DetectionMultiscalePrePredictionCallback
from super_gradients.training.utils.callbacks import DeciLabUploadCallback, LRCallbackBase, LRSchedulerCallback, \
    MetricsUpdateCallback, \
    ModelConversionCheckCallback, YoloXTrainingStageSwitchCallback
from super_gradients.training.utils.early_stopping import EarlyStop


class CallbacksFactory(BaseFactory):

    def __init__(self):
        type_dict = {
            'DeciLabUploadCallback': DeciLabUploadCallback,
            'LRCallbackBase': LRCallbackBase,
            'LRSchedulerCallback': LRSchedulerCallback,
            'MetricsUpdateCallback': MetricsUpdateCallback,
            'ModelConversionCheckCallback': ModelConversionCheckCallback,
            'EarlyStop': EarlyStop,
            'DetectionMultiscalePrePredictionCallback': DetectionMultiscalePrePredictionCallback,
            'YoloXTrainingStageSwitchCallback': YoloXTrainingStageSwitchCallback


        }
        super().__init__(type_dict)
