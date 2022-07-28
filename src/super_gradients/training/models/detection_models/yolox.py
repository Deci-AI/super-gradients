from super_gradients.training.models.detection_models.yolo_base import YoLoBase, YoLoDarknetBackbone
from super_gradients.training.utils.utils import HpmStruct


class YoloX_N(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.25
        arch_params.yolo_type = 'yoloX'
        arch_params.depthwise = True
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_T(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.375
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_S(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.50
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_M(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.67
        arch_params.width_mult_factor = 0.75
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_L(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.0
        arch_params.width_mult_factor = 1.0
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)


class YoloX_X(YoLoBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.33
        arch_params.width_mult_factor = 1.25
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoLoDarknetBackbone, arch_params=arch_params)
