from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.all_target_generators import ALL_TARGET_GENERATORS


class TargetGeneratorsFactory(BaseFactory):
    def __init__(self):
        super().__init__(ALL_TARGET_GENERATORS)
