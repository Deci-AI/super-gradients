from super_gradients.training.models.detection_models.yolov5_base import YoLoV5Base, YoLoV5DarknetBackbone
from super_gradients.training.utils.utils import HpmStruct, get_param


class Custom_YoLoV5(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        backbone = get_param(arch_params, 'backbone', YoLoV5DarknetBackbone)
        super().__init__(backbone=backbone, arch_params=arch_params)


class YoLoV5N(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.25
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5S(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.50
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5M(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.67
        arch_params.width_mult_factor = 0.75
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5L(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.0
        arch_params.width_mult_factor = 1.0
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)


class YoLoV5X(YoLoV5Base):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.33
        arch_params.width_mult_factor = 1.25
        super().__init__(backbone=YoLoV5DarknetBackbone, arch_params=arch_params)
