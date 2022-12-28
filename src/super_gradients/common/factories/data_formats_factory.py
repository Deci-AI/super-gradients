from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.training.utils.tensor_format_adapters.default_formats import DEFAULT_CONCATENATED_TENSOR_FORMATS


class ConcatenatedTensorFormatFactory(TypeFactory):
    def __init__(self):
        super().__init__(DEFAULT_CONCATENATED_TENSOR_FORMATS)
