from super_gradients.modules.pose_estimation_modules import LightweightDEKRHead
from super_gradients.modules.detection_modules import (
    MobileNetV1Backbone,
    MobileNetV2Backbone,
    SSDInvertedResidualNeck,
    SSDBottleneckNeck,
    NStageBackbone,
    PANNeck,
    NHeads,
    SSDHead,
)
from super_gradients.training.models.detection_models.csp_resnet import CSPResNetBackbone
from super_gradients.training.models.detection_models.pp_yolo_e.pan import CustomCSPPAN

ALL_DETECTION_MODULES = {
    "MobileNetV1Backbone": MobileNetV1Backbone,
    "MobileNetV2Backbone": MobileNetV2Backbone,
    "SSDInvertedResidualNeck": SSDInvertedResidualNeck,
    "SSDBottleneckNeck": SSDBottleneckNeck,
    "SSDHead": SSDHead,
    "NStageBackbone": NStageBackbone,
    "PANNeck": PANNeck,
    "NHeads": NHeads,
    "LightweightDEKRHead": LightweightDEKRHead,
    "CustomCSPPAN": CustomCSPPAN,
    "CSPResNetBackbone": CSPResNetBackbone,
}
