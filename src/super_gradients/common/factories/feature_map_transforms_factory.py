from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.transforms.feature_map_transforms import FEATURE_MAP_TRANSFORMS


class FeatureMapTransformsFactory(BaseFactory):
    def __init__(self):
        super().__init__(FEATURE_MAP_TRANSFORMS)
