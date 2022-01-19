from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.utils.callbacks import DeciLabUploadCallback, LRCallbackBase, LRSchedulerCallback, MetricsUpdateCallback, \
    ModelConversionCheckCallback
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

        }
        super().__init__(type_dict)
