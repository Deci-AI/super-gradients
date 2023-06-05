import warnings

from .sg_module import SgModule

# Classification models
from super_gradients.training.models.classification_models.beit import Beit, BeitLargePatch16_224, BeitBasePatch16_224
from super_gradients.training.models.classification_models.densenet import DenseNet, CustomizedDensnet, DenseNet121, DenseNet161, DenseNet169, DenseNet201
from super_gradients.training.models.classification_models.dpn import DPN, DPN26, DPN92
from super_gradients.training.models.classification_models.efficientnet import (
    CustomizedEfficientnet,
    EfficientNet,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetL2,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetB8,
)
from super_gradients.training.models.classification_models.googlenet import GoogleNetV1, GoogLeNet
from super_gradients.training.models.classification_models.lenet import LeNet
from super_gradients.training.models.classification_models.mobilenet import MobileNet
from super_gradients.training.models.classification_models.mobilenetv2 import MobileNetBase, MobileNetV2Base, MobileNetV2, MobileNetV2_135, CustomMobileNetV2
from super_gradients.training.models.classification_models.mobilenetv3 import MobileNetV3
from super_gradients.training.models.classification_models.pnasnet import PNASNet, PNASNetA, PNASNetB
from super_gradients.training.models.classification_models.preact_resnet import (
    PreActResNet,
    PreActResNet18,
    PreActResNet34,
    PreActResNet50,
    PreActResNet101,
    PreActResNet152,
)
from super_gradients.training.models.classification_models.resnet import (
    Bottleneck as NewBottleneck,
    BasicResNetBlock,
    ResNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet18Cifar,
    ResNet50_3343,
    CifarResNet,
    CustomizedResnet,
    CustomizedResnet50Cifar,
    CustomizedResnetCifar,
    CustomizedResnet50,
)
from super_gradients.training.models.classification_models.resnext import ResNeXt, ResNeXt50, ResNeXt101
from super_gradients.training.models.classification_models.senet import SENet, SENet18
from super_gradients.training.models.classification_models.shufflenet import ShuffleNet, ShuffleNetG2, ShuffleNetG3
from super_gradients.training.models.classification_models.shufflenetv2 import (
    ShuffleNetV2Base,
    ShufflenetV2_x0_5,
    ShufflenetV2_x1_0,
    ShufflenetV2_x1_5,
    CustomizedShuffleNetV2,
    ShufflenetV2_x2_0,
)
from super_gradients.training.models.classification_models.vgg import VGG
from super_gradients.training.models.classification_models.vit import ViT, ViTBase, ViTLarge, ViTHuge

# Detection models
from super_gradients.training.models.detection_models.csp_darknet53 import CSPDarknet53, SPP
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloE, PPYoloE_S, PPYoloE_M, PPYoloE_L, PPYoloE_X
from super_gradients.training.models.detection_models.darknet53 import Darknet53, Darknet53Base
from super_gradients.training.models.detection_models.ssd import SSDMobileNetV1, SSDLiteMobileNetV2
from super_gradients.training.models.detection_models.yolo_base import YoloBase, YoloPostPredictionCallback
from super_gradients.training.models.detection_models.yolox import YoloX_N, YoloX_T, YoloX_S, YoloX_M, YoloX_L, YoloX_X, CustomYoloX
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.models.detection_models.yolo_nas import (
    YoloNASStage,
    YoloNASStem,
    YoloNASDownStage,
    YoloNASUpStage,
    YoloNASBottleneck,
    YoloNASDFLHead,
    NDFLHeads,
    YoloNASPANNeckWithC2,
    YoloNAS_S,
    YoloNAS_M,
    YoloNAS_L,
)

# Segmentation models
from super_gradients.training.models.segmentation_models.shelfnet import (
    ShelfNet50,
    ShelfNet101,
    ShelfNetHW,
    ShelfNetLW,
    ShelfNetBase,
    ShelfNet503343,
    ShelfNet18_LW,
    ShelfNet34_LW,
)
from super_gradients.training.models.segmentation_models.unet.unet import UNet, UNetBase, UNetCustom
from super_gradients.training.models.segmentation_models.ddrnet import DDRNet, DDRNet23, DDRNet39, DDRNetCustom, DDRNet23Slim
from super_gradients.training.models.segmentation_models.laddernet import LadderNet, LadderNet50, LadderNet101, LadderNet503433
from super_gradients.training.models.segmentation_models.ppliteseg import PPLiteSegBase, PPLiteSegB, PPLiteSegT
from super_gradients.training.models.segmentation_models.regseg import RegSeg, RegSeg48, RegSeg53
from super_gradients.training.models.segmentation_models.stdc import (
    STDC1Seg,
    STDC2Seg,
    STDCClassification,
    STDC1Classification,
    STDCClassificationBase,
    STDC2Classification,
    STDCSegmentationBase,
    CustomSTDCSegmentation,
)
from super_gradients.training.models.segmentation_models.segformer import SegFormerB0, SegFormerB1, SegFormerB2, SegFormerB3, SegFormerB4, SegFormerB5
from super_gradients.training.models.segmentation_models.ddrnet_backbones import DDRNet39Backbone

# Pose estimation
from super_gradients.training.models.pose_estimation_models.dekr_hrnet import DEKRPoseEstimationModel, DEKRW32NODC

# KD
from super_gradients.training.models.kd_modules.kd_module import KDModule

import super_gradients.training.models.user_models as user_models
from super_gradients.training.models.model_factory import get, get_model_name
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.conversion import convert_to_coreml, convert_to_onnx, convert_from_config


from super_gradients.common.object_names import Models
from super_gradients.common.registry.registry import ARCHITECTURES

from super_gradients.training.utils import make_divisible as _make_divisible_current_version, HpmStruct as CurrVersionHpmStruct


def make_deprecated(func, reason):
    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("once", DeprecationWarning)
            warnings.warn(reason, category=DeprecationWarning, stacklevel=2)
        warnings.warn(reason, DeprecationWarning)
        return func(*args, **kwargs)

    return inner


make_divisible = make_deprecated(
    func=_make_divisible_current_version,
    reason="You're importing `make_divisible` from `super_gradients.training.models`. This is deprecated since SuperGradients 3.1.0.\n"
    "Please update your code to import it as follows:\n"
    "[-] from super_gradients.training.models import make_divisible\n"
    "[+] from super_gradients.training.utils import make_divisible\n",
)


BasicBlock = make_deprecated(
    func=BasicResNetBlock,
    reason="You're importing `BasicBlock` class from `super_gradients.training.models`. This is deprecated since SuperGradients 3.1.0.\n"
    "This block was renamed to BasicResNetBlock for better clarity.\n"
    "Please update your code to import it as follows:\n"
    "[-] from super_gradients.training.models import BasicBlock\n"
    "[+] from super_gradients.training.models import BasicResNetBlock\n",
)

Bottleneck = make_deprecated(
    func=NewBottleneck,
    reason="You're importing `Bottleneck` class from `super_gradients.training.models`. This is deprecated since SuperGradients 3.1.0.\n"
    "This block was renamed to BasicResNetBlock for better clarity.\n"
    "Please update your code to import it as follows:\n"
    "[-] from super_gradients.training.models import Bottleneck\n"
    "[+] from super_gradients.training.models.classification_models.resnet import Bottleneck\n",
)

HpmStruct = make_deprecated(
    func=CurrVersionHpmStruct,
    reason="You're importing `HpmStruct` class from `super_gradients.training.models`. This is deprecated since SuperGradients 3.1.0.\n"
    "Please update your code to import it as follows:\n"
    "[-] from super_gradients.training.models import HpmStruct\n"
    "[+] from super_gradients.training.utils import HpmStruct\n",
)


__all__ = [
    "HpmStruct",
    "Bottleneck",
    "SPP",
    "YoloNAS_S",
    "YoloNAS_M",
    "YoloNAS_L",
    "YoloNASStage",
    "YoloNASUpStage",
    "YoloNASStem",
    "YoloNASDownStage",
    "YoloNASDFLHead",
    "YoloNASBottleneck",
    "NDFLHeads",
    "YoloNASPANNeckWithC2",
    "SgModule",
    "Beit",
    "BeitLargePatch16_224",
    "BeitBasePatch16_224",
    "DenseNet",
    "CustomizedDensnet",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
    "DPN",
    "DPN26",
    "DPN92",
    "CustomizedEfficientnet",
    "EfficientNet",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetL2",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
    "EfficientNetB8",
    "GoogleNetV1",
    "GoogLeNet",
    "LeNet",
    "MobileNet",
    "MobileNetBase",
    "MobileNetV2Base",
    "MobileNetV2",
    "MobileNetV2_135",
    "CustomMobileNetV2",
    "MobileNetV3",
    "PNASNet",
    "PNASNetA",
    "PNASNetB",
    "PreActResNet",
    "PreActResNet18",
    "PreActResNet34",
    "PreActResNet50",
    "PreActResNet101",
    "PreActResNet152",
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet18Cifar",
    "ResNet50_3343",
    "CifarResNet",
    "CustomizedResnet",
    "CustomizedResnet50Cifar",
    "CustomizedResnetCifar",
    "CustomizedResnet50",
    "ResNeXt",
    "ResNeXt50",
    "ResNeXt101",
    "SENet",
    "SENet18",
    "ShuffleNet",
    "ShuffleNetG2",
    "ShuffleNetG3",
    "ShuffleNetV2Base",
    "ShufflenetV2_x0_5",
    "ShufflenetV2_x1_0",
    "ShufflenetV2_x1_5",
    "CustomizedShuffleNetV2",
    "ShufflenetV2_x2_0",
    "VGG",
    "ViT",
    "ViTBase",
    "ViTLarge",
    "ViTHuge",
    "CSPDarknet53",
    "PPYoloE",
    "PPYoloE_S",
    "PPYoloE_M",
    "PPYoloE_L",
    "PPYoloE_X",
    "Darknet53",
    "Darknet53Base",
    "SSDMobileNetV1",
    "SSDLiteMobileNetV2",
    "YoloBase",
    "YoloX_N",
    "YoloX_T",
    "YoloX_S",
    "YoloX_M",
    "YoloX_L",
    "YoloX_X",
    "CustomYoloX",
    "YoloPostPredictionCallback",
    "CustomizableDetector",
    "ShelfNet50",
    "ShelfNet101",
    "ShelfNetHW",
    "ShelfNetLW",
    "ShelfNetBase",
    "ShelfNet503343",
    "ShelfNet18_LW",
    "ShelfNet34_LW",
    "UNet",
    "UNetBase",
    "UNetCustom",
    "DDRNet",
    "DDRNet23",
    "DDRNet39",
    "DDRNetCustom",
    "DDRNet23Slim",
    "LadderNet",
    "LadderNet50",
    "LadderNet101",
    "LadderNet503433",
    "PPLiteSegBase",
    "PPLiteSegB",
    "PPLiteSegT",
    "RegSeg",
    "RegSeg48",
    "RegSeg53",
    "STDC1Seg",
    "STDC2Seg",
    "STDCClassification",
    "STDC1Classification",
    "STDCClassificationBase",
    "STDC2Classification",
    "STDCSegmentationBase",
    "CustomSTDCSegmentation",
    "DEKRPoseEstimationModel",
    "DEKRW32NODC",
    "KDModule",
    "get",
    "get_model_name",
    "get_arch_params",
    "convert_to_coreml",
    "convert_to_onnx",
    "convert_from_config",
    "ARCHITECTURES",
    "Models",
    "user_models",
    "SegFormerB0",
    "SegFormerB1",
    "SegFormerB2",
    "SegFormerB3",
    "SegFormerB4",
    "SegFormerB5",
    "DDRNet39Backbone",
    "make_divisible",
    "BasicResNetBlock",
    "BasicBlock",
]
