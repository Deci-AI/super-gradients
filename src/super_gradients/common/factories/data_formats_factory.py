from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.training.utils.output_adapters.default_formats import DEFAULT_DATA_FORMATS


class DataFormatFactory(TypeFactory):
    def __init__(self):
        super().__init__(DEFAULT_DATA_FORMATS)
