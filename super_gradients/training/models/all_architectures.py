from super_gradients.training.models import ResNeXt50, ResNeXt101, googlenet_v1
from super_gradients.training.models.classification_models import repvgg, efficientnet, densenet, resnet, regnet
from super_gradients.training.models.classification_models.mobilenetv2 import mobile_net_v2, mobile_net_v2_135, \
    custom_mobile_net_v2
from super_gradients.training.models.classification_models.mobilenetv3 import mobilenetv3_large, mobilenetv3_small, \
    mobilenetv3_custom
from super_gradients.training.models.classification_models.shufflenetv2 import ShufflenetV2_x0_5, ShufflenetV2_x1_0, \
    ShufflenetV2_x1_5, \
    ShufflenetV2_x2_0, CustomizedShuffleNetV2
from super_gradients.training.models.detection_models.csp_darknet53 import CSPDarknet53
from super_gradients.training.models.detection_models.darknet53 import Darknet53
from super_gradients.training.models.detection_models.ssd import SSDMobileNetV1, SSDLiteMobileNetV2
from super_gradients.training.models.detection_models.yolov3 import YoloV3, TinyYoloV3
from super_gradients.training.models.detection_models.yolov5 import YoLoV5N, YoLoV5S, YoLoV5M, YoLoV5L, YoLoV5X, Custom_YoLoV5
from super_gradients.training.models.segmentation_models.ddrnet import DDRNet23, DDRNet23Slim, AnyBackBoneDDRNet23
from super_gradients.training.models.segmentation_models.shelfnet import ShelfNet18_LW, ShelfNet34_LW, ShelfNet50, \
    ShelfNet503343, ShelfNet101

# IF YOU UPDATE THE ARCHITECTURE DICT PLEASE ALSO UPDATE THE ENUM CLASS DOWN BELOW.


ARCHITECTURES = {"resnet18": resnet.ResNet18,
                 "resnet34": resnet.ResNet34,
                 "resnet50_3343": resnet.ResNet50_3343,
                 "resnet50": resnet.ResNet50,
                 "resnet101": resnet.ResNet101,
                 "resnet152": resnet.ResNet152,
                 "resnet18_cifar": resnet.ResNet18Cifar,
                 "custom_resnet": resnet.CustomizedResnet,
                 "custom_resnet50": resnet.CustomizedResnet50,
                 "custom_resnet_cifar": resnet.CustomizedResnetCifar,
                 "custom_resnet50_cifar": resnet.CustomizedResnet50Cifar,
                 "mobilenet_v2": mobile_net_v2,
                 "mobile_net_v2_135": mobile_net_v2_135,
                 "custom_mobilenet_v2": custom_mobile_net_v2,
                 "mobilenet_v3_large": mobilenetv3_large,
                 "mobilenet_v3_small": mobilenetv3_small,
                 "mobilenet_v3_custom": mobilenetv3_custom,
                 "yolo_v3": YoloV3,
                 "tiny_yolo_v3": TinyYoloV3,
                 "custom_densenet": densenet.CustomizedDensnet,
                 "densenet121": densenet.densenet121,
                 "densenet161": densenet.densenet161,
                 "densenet169": densenet.densenet169,
                 "densenet201": densenet.densenet201,
                 "shelfnet18_lw": ShelfNet18_LW,
                 "shelfnet34_lw": ShelfNet34_LW,
                 "shelfnet50_3343": ShelfNet503343,
                 "shelfnet50": ShelfNet50,
                 "shelfnet101": ShelfNet101,
                 "shufflenet_v2_x0_5": ShufflenetV2_x0_5,
                 "shufflenet_v2_x1_0": ShufflenetV2_x1_0,
                 "shufflenet_v2_x1_5": ShufflenetV2_x1_5,
                 "shufflenet_v2_x2_0": ShufflenetV2_x2_0,
                 "shufflenet_v2_custom5": CustomizedShuffleNetV2,
                 'darknet53': Darknet53,
                 'csp_darknet53': CSPDarknet53,
                 "resnext50": ResNeXt50,
                 "resnext101": ResNeXt101,
                 "googlenet_v1": googlenet_v1,
                 "efficientnet_b0": efficientnet.b0,
                 "efficientnet_b1": efficientnet.b1,
                 "efficientnet_b2": efficientnet.b2,
                 "efficientnet_b3": efficientnet.b3,
                 "efficientnet_b4": efficientnet.b4,
                 "efficientnet_b5": efficientnet.b5,
                 "efficientnet_b6": efficientnet.b6,
                 "efficientnet_b7": efficientnet.b7,
                 "efficientnet_b8": efficientnet.b8,
                 "efficientnet_l2": efficientnet.l2,
                 "CustomizedEfficientnet": efficientnet.CustomizedEfficientnet,
                 "regnetY200": regnet.RegNetY200,
                 "regnetY400": regnet.RegNetY400,
                 "regnetY600": regnet.RegNetY600,
                 "regnetY800": regnet.RegNetY800,
                 "custom_regnet": regnet.CustomRegNet,
                 "nas_regnet": regnet.NASRegNet,
                 "yolo_v5n": YoLoV5N,
                 "yolo_v5s": YoLoV5S,
                 "yolo_v5m": YoLoV5M,
                 "yolo_v5l": YoLoV5L,
                 "yolo_v5x": YoLoV5X,
                 "custom_yolov5": Custom_YoLoV5,
                 "ssd_mobilenet_v1": SSDMobileNetV1,
                 "ssd_lite_mobilenet_v2": SSDLiteMobileNetV2,
                 "repvgg_a0": repvgg.RepVggA0,
                 "repvgg_a1": repvgg.RepVggA1,
                 "repvgg_a2": repvgg.RepVggA2,
                 "repvgg_b0": repvgg.RepVggB0,
                 "repvgg_b1": repvgg.RepVggB1,
                 "repvgg_b2": repvgg.RepVggB2,
                 "repvgg_b3": repvgg.RepVggB3,
                 "repvgg_d2se": repvgg.RepVggD2SE,
                 "repvgg_custom": repvgg.RepVggCustom,
                 "ddrnet_23": DDRNet23,
                 "ddrnet_23_slim": DDRNet23Slim,
                 "custom_ddrnet_23": AnyBackBoneDDRNet23,
                 }
