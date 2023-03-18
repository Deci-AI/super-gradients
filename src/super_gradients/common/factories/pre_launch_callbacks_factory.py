from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.registry import ALL_PRE_LAUNCH_CALLBACKS


class PreLaunchCallbacksFactory(BaseFactory):
    def __init__(self):
        super().__init__(ALL_PRE_LAUNCH_CALLBACKS)
