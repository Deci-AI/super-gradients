from super_gradients.training.models import ResNeXt50, ResNeXt101, GoogleNetV1
from super_gradients.training.models.classification_models import repvgg, efficientnet, densenet, resnet, regnet
from super_gradients.training.models.classification_models.mobilenetv2 import MobileNetV2Base, MobileNetV2_135, CustomMobileNetV2
from super_gradients.training.models.classification_models.mobilenetv3 import mobilenetv3_large, mobilenetv3_small, mobilenetv3_custom
from super_gradients.training.models.classification_models.shufflenetv2 import (
    ShufflenetV2_x0_5,
    ShufflenetV2_x1_0,
    ShufflenetV2_x1_5,
    ShufflenetV2_x2_0,
    CustomizedShuffleNetV2,
)
from super_gradients.training.models.classification_models.vit import ViTBase, ViTLarge, ViTHuge
from super_gradients.training.models.detection_models.csp_darknet53 import CSPDarknet53
from super_gradients.training.models.detection_models.darknet53 import Darknet53
from super_gradients.training.models.detection_models.ssd import SSDMobileNetV1, SSDLiteMobileNetV2
from super_gradients.training.models.detection_models.yolox import YoloX_N, YoloX_T, YoloX_S, YoloX_M, YoloX_L, YoloX_X, CustomYoloX
from super_gradients.training.models.segmentation_models.ddrnet import DDRNet23, DDRNet23Slim, AnyBackBoneDDRNet23
from super_gradients.training.models.segmentation_models.regseg import RegSeg48
from super_gradients.training.models.segmentation_models.shelfnet import ShelfNet18_LW, ShelfNet34_LW, ShelfNet50, ShelfNet503343, ShelfNet101
from super_gradients.training.models.segmentation_models.stdc import STDC1Classification, STDC2Classification, STDC1Seg, STDC2Seg, STDCSegmentationBase

from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.models.classification_models.beit import BeitBasePatch16_224, BeitLargePatch16_224
from super_gradients.training.models.segmentation_models.ppliteseg import PPLiteSegT, PPLiteSegB
from super_gradients.training.models.segmentation_models.unet import UNetCustom, UnetClassification
from super_gradients.common.object_names import Models

ARCHITECTURES = {
    Models.RESNET18: resnet.ResNet18,
    Models.RESNET34: resnet.ResNet34,
    Models.RESNET50_3343: resnet.ResNet50_3343,
    Models.RESNET50: resnet.ResNet50,
    Models.RESNET101: resnet.ResNet101,
    Models.RESNET152: resnet.ResNet152,
    Models.RESNET18_CIFAR: resnet.ResNet18Cifar,
    Models.CUSTOM_RESNET: resnet.CustomizedResnet,
    Models.CUSTOM_RESNET50: resnet.CustomizedResnet50,
    Models.CUSTOM_RESNET_CIFAR: resnet.CustomizedResnetCifar,
    Models.CUSTOM_RESNET50_CIFAR: resnet.CustomizedResnet50Cifar,
    Models.MOBILENET_V2: MobileNetV2Base,
    Models.MOBILE_NET_V2_135: MobileNetV2_135,
    Models.CUSTOM_MOBILENET_V2: CustomMobileNetV2,
    Models.MOBILENET_V3_LARGE: mobilenetv3_large,
    Models.MOBILENET_V3_SMALL: mobilenetv3_small,
    Models.MOBILENET_V3_CUSTOM: mobilenetv3_custom,
    Models.CUSTOM_DENSENET: densenet.CustomizedDensnet,
    Models.DENSENET121: densenet.DenseNet121,
    Models.DENSENET161: densenet.DenseNet161,
    Models.DENSENET169: densenet.DenseNet169,
    Models.DENSENET201: densenet.DenseNet201,
    Models.SHELFNET18_LW: ShelfNet18_LW,
    Models.SHELFNET34_LW: ShelfNet34_LW,
    Models.SHELFNET50_3343: ShelfNet503343,
    Models.SHELFNET50: ShelfNet50,
    Models.SHELFNET101: ShelfNet101,
    Models.SHUFFLENET_V2_X0_5: ShufflenetV2_x0_5,
    Models.SHUFFLENET_V2_X1_0: ShufflenetV2_x1_0,
    Models.SHUFFLENET_V2_X1_5: ShufflenetV2_x1_5,
    Models.SHUFFLENET_V2_X2_0: ShufflenetV2_x2_0,
    Models.SHUFFLENET_V2_CUSTOM5: CustomizedShuffleNetV2,
    Models.DARKNET53: Darknet53,
    Models.CSP_DARKNET53: CSPDarknet53,
    Models.RESNEXT50: ResNeXt50,
    Models.RESNEXT101: ResNeXt101,
    Models.GOOGLENET_V1: GoogleNetV1,
    Models.EFFICIENTNET_B0: efficientnet.EfficientNetB0,
    Models.EFFICIENTNET_B1: efficientnet.EfficientNetB1,
    Models.EFFICIENTNET_B2: efficientnet.EfficientNetB2,
    Models.EFFICIENTNET_B3: efficientnet.EfficientNetB3,
    Models.EFFICIENTNET_B4: efficientnet.EfficientNetB4,
    Models.EFFICIENTNET_B5: efficientnet.EfficientNetB5,
    Models.EFFICIENTNET_B6: efficientnet.EfficientNetB6,
    Models.EFFICIENTNET_B7: efficientnet.EfficientNetB7,
    Models.EFFICIENTNET_B8: efficientnet.EfficientNetB8,
    Models.EFFICIENTNET_L2: efficientnet.EfficientNetL2,
    Models.CUSTOMIZEDEFFICIENTNET: efficientnet.CustomizedEfficientnet,
    Models.REGNETY200: regnet.RegNetY200,
    Models.REGNETY400: regnet.RegNetY400,
    Models.REGNETY600: regnet.RegNetY600,
    Models.REGNETY800: regnet.RegNetY800,
    Models.CUSTOM_REGNET: regnet.CustomRegNet,
    Models.NAS_REGNET: regnet.NASRegNet,
    Models.YOLOX_N: YoloX_N,
    Models.YOLOX_T: YoloX_T,
    Models.YOLOX_S: YoloX_S,
    Models.YOLOX_M: YoloX_M,
    Models.YOLOX_L: YoloX_L,
    Models.YOLOX_X: YoloX_X,
    Models.CUSTOM_YOLO_X: CustomYoloX,
    Models.SSD_MOBILENET_V1: SSDMobileNetV1,
    Models.SSD_LITE_MOBILENET_V2: SSDLiteMobileNetV2,
    Models.REPVGG_A0: repvgg.RepVggA0,
    Models.REPVGG_A1: repvgg.RepVggA1,
    Models.REPVGG_A2: repvgg.RepVggA2,
    Models.REPVGG_B0: repvgg.RepVggB0,
    Models.REPVGG_B1: repvgg.RepVggB1,
    Models.REPVGG_B2: repvgg.RepVggB2,
    Models.REPVGG_B3: repvgg.RepVggB3,
    Models.REPVGG_D2SE: repvgg.RepVggD2SE,
    Models.REPVGG_CUSTOM: repvgg.RepVggCustom,
    Models.DDRNET_23: DDRNet23,
    Models.DDRNET_23_SLIM: DDRNet23Slim,
    Models.CUSTOM_DDRNET_23: AnyBackBoneDDRNet23,
    Models.STDC1_CLASSIFICATION: STDC1Classification,
    Models.STDC2_CLASSIFICATION: STDC2Classification,
    Models.STDC1_SEG: STDC1Seg,
    Models.STDC1_SEG50: STDC1Seg,
    Models.STDC1_SEG75: STDC1Seg,
    Models.STDC2_SEG: STDC2Seg,
    Models.STDC2_SEG50: STDC2Seg,
    Models.STDC2_SEG75: STDC2Seg,
    Models.CUSTOM_STDC: STDCSegmentationBase,
    Models.REGSEG48: RegSeg48,
    Models.KD_MODULE: KDModule,
    Models.VIT_BASE: ViTBase,
    Models.VIT_LARGE: ViTLarge,
    Models.VIT_HUGE: ViTHuge,
    Models.BEIT_BASE_PATCH16_224: BeitBasePatch16_224,
    Models.BEIT_LARGE_PATCH16_224: BeitLargePatch16_224,
    Models.PP_LITE_T_SEG: PPLiteSegT,
    Models.PP_LITE_T_SEG50: PPLiteSegT,
    Models.PP_LITE_T_SEG75: PPLiteSegT,
    Models.PP_LITE_B_SEG: PPLiteSegB,
    Models.PP_LITE_B_SEG50: PPLiteSegB,
    Models.PP_LITE_B_SEG75: PPLiteSegB,
    Models.CUSTOM_ANYNET: regnet.CustomAnyNet,
    Models.UNET_CUSTOM: UNetCustom,
    Models.UNET_CUSTOM_CLS: UnetClassification,
}

KD_ARCHITECTURES = {Models.KD_MODULE: KDModule}
