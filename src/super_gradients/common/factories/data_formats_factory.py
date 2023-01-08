from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.training.datasets.data_formats.default_formats import DEFAULT_CONCATENATED_TENSOR_FORMATS


class ConcatenatedTensorFormatFactory(TypeFactory):
    def __init__(self):
        super().__init__(DEFAULT_CONCATENATED_TENSOR_FORMATS)
