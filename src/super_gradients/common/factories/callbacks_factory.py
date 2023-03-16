from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.registry import CALLBACKS


class CallbacksFactory(BaseFactory):
    def __init__(self):
        super().__init__(CALLBACKS)
