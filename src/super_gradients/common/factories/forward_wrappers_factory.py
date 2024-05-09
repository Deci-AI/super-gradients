from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.registry import FORWARD_WRAPPERS


class ForwardWrappersFactory(BaseFactory):
    def __init__(self):
        super().__init__(FORWARD_WRAPPERS)
