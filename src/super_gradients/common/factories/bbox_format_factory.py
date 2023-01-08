from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.data_formats.bbox_formats import BBOX_FORMATS


class BBoxFormatFactory(BaseFactory):
    def __init__(self):
        super().__init__(BBOX_FORMATS)
