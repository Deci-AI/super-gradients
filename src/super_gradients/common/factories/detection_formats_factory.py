from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.training.utils.output_adapters.default_formats import DEFAULT_DETECTION_FORMATS


class OptimizersTypeFactory(TypeFactory):
    def __init__(self):
        super().__init__(DEFAULT_DETECTION_FORMATS)
