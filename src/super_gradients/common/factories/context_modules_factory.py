from super_gradients.common.factories.base_factory import BaseFactory
import super_gradients.training.models.segmentation_models.context_modules as context_modules


class ContextModulesFactory(BaseFactory):
    def __init__(self):
        super().__init__(context_modules.CONTEXT_TYPE_DICT)
