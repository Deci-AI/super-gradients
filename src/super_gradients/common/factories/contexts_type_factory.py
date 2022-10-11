from super_gradients.common.factories import TypeFactory
import super_gradients.training.models.segmentation_models.context_modules as context_modules


class ContextsTypeFactory(TypeFactory):
    def __init__(self):
        super().__init__(context_modules.CONTEXT_TYPE_DICT)
