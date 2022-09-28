from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.metrics import METRICS


class MetricsFactory(BaseFactory):

    def __init__(self):
        super().__init__(METRICS)
