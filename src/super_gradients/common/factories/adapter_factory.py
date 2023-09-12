from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.registry import DATASET_ADAPTERS


class AdaptersFactory(BaseFactory):
    def __init__(self):
        super().__init__(DATASET_ADAPTERS)
