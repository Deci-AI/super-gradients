from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training import pre_launch_callbacks


class PreLaunchCallbacksFactory(BaseFactory):
    def __init__(self):
        super().__init__(pre_launch_callbacks.ALL_PRE_LAUNCH_CALLBACKS)
