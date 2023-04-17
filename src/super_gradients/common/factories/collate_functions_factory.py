from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.registry import ALL_COLLATE_FUNCTIONS


class CollateFunctionsFactory(BaseFactory):
    def __init__(self):
        super().__init__(ALL_COLLATE_FUNCTIONS)
