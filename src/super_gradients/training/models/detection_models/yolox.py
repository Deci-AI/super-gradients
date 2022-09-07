from super_gradients.training.models.detection_models.yolo_base import YoloBase, YoloDarknetBackbone
from super_gradients.training.utils.utils import HpmStruct


class YoloX_N(YoloBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.25
        arch_params.yolo_type = 'yoloX'
        arch_params.depthwise = True
        super().__init__(backbone=YoloDarknetBackbone, arch_params=arch_params)


class YoloX_T(YoloBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.375
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoloDarknetBackbone, arch_params=arch_params)


class YoloX_S(YoloBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.33
        arch_params.width_mult_factor = 0.50
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoloDarknetBackbone, arch_params=arch_params)


class YoloX_M(YoloBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 0.67
        arch_params.width_mult_factor = 0.75
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoloDarknetBackbone, arch_params=arch_params)


class YoloX_L(YoloBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.0
        arch_params.width_mult_factor = 1.0
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoloDarknetBackbone, arch_params=arch_params)


class YoloX_X(YoloBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.depth_mult_factor = 1.33
        arch_params.width_mult_factor = 1.25
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=YoloDarknetBackbone, arch_params=arch_params)


class CustomYoloX(YoloBase):
    def __init__(self, arch_params: HpmStruct):
        arch_params.yolo_type = 'yoloX'
        super().__init__(backbone=arch_params.backbone, arch_params=arch_params)
