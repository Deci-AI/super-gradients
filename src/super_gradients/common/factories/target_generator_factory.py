from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.target_generator import ALL_TARGET_GENERATORS


class TargetGeneratorsFactory(BaseFactory):
    def __init__(self):
        super().__init__(ALL_TARGET_GENERATORS)
