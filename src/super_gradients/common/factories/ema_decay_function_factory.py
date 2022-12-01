from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.utils.ema_decay_schedules import EMA_DECAY_FUNCTIONS


class EMADecayFunctionFactory(BaseFactory):
    def __init__(self):
        super().__init__(EMA_DECAY_FUNCTIONS)
