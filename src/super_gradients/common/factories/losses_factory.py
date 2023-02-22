from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.register_loss import loss_registry


class LossesFactory(BaseFactory):
    def __init__(self):
        super().__init__(loss_registry.items)
