from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.samplers import SAMPLERS


class SamplersFactory(BaseFactory):

    def __init__(self):
        super().__init__(SAMPLERS)
