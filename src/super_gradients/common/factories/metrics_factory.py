from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.metrics import Accuracy, Top5, DetectionMetrics, IoU, PixelAccuracy


class MetricsFactory(BaseFactory):

    def __init__(self):
        type_dict = {
            'Accuracy': Accuracy,
            'Top5': Top5,
            'DetectionMetrics': DetectionMetrics,
            'IoU': IoU,
            'PixelAccuracy': PixelAccuracy,
        }
        super().__init__(type_dict)
