from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.training.utils.collate_fn.all_collate_fn import COLLATE_FUNCTIONS


class ConcatenatedTensorFormatFactory(TypeFactory):
    def __init__(self):
        super().__init__(COLLATE_FUNCTIONS)
