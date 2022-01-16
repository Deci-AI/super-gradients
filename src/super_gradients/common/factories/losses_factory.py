from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.losses import LOSSES


class LossesFactory(BaseFactory):

    def __init__(self):
        super().__init__(LOSSES)
