from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.utils.callbacks import CALLBACKS


class CallbacksFactory(BaseFactory):

    def __init__(self):
        super().__init__(CALLBACKS)
