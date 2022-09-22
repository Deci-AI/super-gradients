from super_gradients.training.models import ResNeXt50, ResNeXt101, GoogleNetV1
from super_gradients.training.models.classification_models import repvgg, efficientnet, densenet, resnet, regnet
from super_gradients.training.models.classification_models.mobilenetv2 import MobileNetV2Base, MobileNetV2_135, \
    CustomMobileNetV2
from super_gradients.training.models.classification_models.mobilenetv3 import mobilenetv3_large, mobilenetv3_small, \
    mobilenetv3_custom
from super_gradients.training.models.classification_models.shufflenetv2 import ShufflenetV2_x0_5, ShufflenetV2_x1_0, \
    ShufflenetV2_x1_5, \
    ShufflenetV2_x2_0, CustomizedShuffleNetV2
from super_gradients.training.models.classification_models.vit import ViTBase, ViTLarge, ViTHuge
from super_gradients.training.models.detection_models.csp_darknet53 import CSPDarknet53
from super_gradients.training.models.detection_models.darknet53 import Darknet53
from super_gradients.training.models.detection_models.ssd import SSDMobileNetV1, SSDLiteMobileNetV2
from super_gradients.training.models.detection_models.yolox import YoloX_N, YoloX_T, YoloX_S, YoloX_M, YoloX_L, YoloX_X, CustomYoloX
from super_gradients.training.models.segmentation_models.ddrnet import DDRNet23, DDRNet23Slim, AnyBackBoneDDRNet23
from super_gradients.training.models.segmentation_models.regseg import RegSeg48
from super_gradients.training.models.segmentation_models.shelfnet import ShelfNet18_LW, ShelfNet34_LW, ShelfNet50, \
    ShelfNet503343, ShelfNet101
from super_gradients.training.models.segmentation_models.stdc import STDC1Classification, STDC2Classification, \
    STDC1Seg, STDC2Seg

from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.models.classification_models.beit import BeitBasePatch16_224, BeitLargePatch16_224
from super_gradients.training.models.segmentation_models.ppliteseg import PPLiteSegT, PPLiteSegB


class ModelNames:
    """Static class to hold all the available model names"""""
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50_3343 = "resnet50_3343"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"
    RESNET18_CIFAR = "resnet18_cifar"
    CUSTOM_RESNET = "custom_resnet"
    CUSTOM_RESNET50 = "custom_resnet50"
    CUSTOM_RESNET_CIFAR = "custom_resnet_cifar"
    CUSTOM_RESNET50_CIFAR = "custom_resnet50_cifar"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILE_NET_V2_135 = "mobile_net_v2_135"
    CUSTOM_MOBILENET_V2 = "custom_mobilenet_v2"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V3_CUSTOM = "mobilenet_v3_custom"
    CUSTOM_DENSENET = "custom_densenet"
    DENSENET121 = "densenet121"
    DENSENET161 = "densenet161"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"
    SHELFNET18_LW = "shelfnet18_lw"
    SHELFNET34_LW = "shelfnet34_lw"
    SHELFNET50_3343 = "shelfnet50_3343"
    SHELFNET50 = "shelfnet50"
    SHELFNET101 = "shelfnet101"
    SHUFFLENET_V2_X0_5 = "shufflenet_v2_x0_5"
    SHUFFLENET_V2_X1_0 = "shufflenet_v2_x1_0"
    SHUFFLENET_V2_X1_5 = "shufflenet_v2_x1_5"
    SHUFFLENET_V2_X2_0 = "shufflenet_v2_x2_0"
    SHUFFLENET_V2_CUSTOM5 = "shufflenet_v2_custom5"
    DARKNET53 = "darknet53"
    CSP_DARKNET53 = "csp_darknet53"
    RESNEXT50 = "resnext50"
    RESNEXT101 = "resnext101"
    GOOGLENET_V1 = "googlenet_v1"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"
    EFFICIENTNET_B4 = "efficientnet_b4"
    EFFICIENTNET_B5 = "efficientnet_b5"
    EFFICIENTNET_B6 = "efficientnet_b6"
    EFFICIENTNET_B7 = "efficientnet_b7"
    EFFICIENTNET_B8 = "efficientnet_b8"
    EFFICIENTNET_L2 = "efficientnet_l2"
    CUSTOMIZEDEFFICIENTNET = "CustomizedEfficientnet"
    REGNETY200 = "regnetY200"
    REGNETY400 = "regnetY400"
    REGNETY600 = "regnetY600"
    REGNETY800 = "regnetY800"
    CUSTOM_REGNET = "custom_regnet"
    NAS_REGNET = "nas_regnet"
    YOLOX_N = "yolox_n"
    YOLOX_T = "yolox_t"
    YOLOX_S = "yolox_s"
    YOLOX_M = "yolox_m"
    YOLOX_L = "yolox_l"
    YOLOX_X = "yolox_x"
    CUSTOM_YOLO_X = "CustomYoloX"
    SSD_MOBILENET_V1 = "ssd_mobilenet_v1"
    SSD_LITE_MOBILENET_V2 = "ssd_lite_mobilenet_v2"
    REPVGG_A0 = "repvgg_a0"
    REPVGG_A1 = "repvgg_a1"
    REPVGG_A2 = "repvgg_a2"
    REPVGG_B0 = "repvgg_b0"
    REPVGG_B1 = "repvgg_b1"
    REPVGG_B2 = "repvgg_b2"
    REPVGG_B3 = "repvgg_b3"
    REPVGG_D2SE = "repvgg_d2se"
    REPVGG_CUSTOM = "repvgg_custom"
    DDRNET_23 = "ddrnet_23"
    DDRNET_23_SLIM = "ddrnet_23_slim"
    CUSTOM_DDRNET_23 = "custom_ddrnet_23"
    STDC1_CLASSIFICATION = "stdc1_classification"
    STDC2_CLASSIFICATION = "stdc2_classification"
    STDC1_SEG = "stdc1_seg"
    STDC1_SEG50 = "stdc1_seg50"
    STDC1_SEG75 = "stdc1_seg75"
    STDC2_SEG = "stdc2_seg"
    STDC2_SEG50 = "stdc2_seg50"
    STDC2_SEG75 = "stdc2_seg75"
    REGSEG48 = "regseg48"
    KD_MODULE = "kd_module"
    VIT_BASE = "vit_base"
    VIT_LARGE = "vit_large"
    VIT_HUGE = "vit_huge"
    BEIT_BASE_PATCH16_224 = "beit_base_patch16_224"
    BEIT_LARGE_PATCH16_224 = "beit_large_patch16_224"
    PP_LITE_T_SEG = "pp_lite_t_seg"
    PP_LITE_T_SEG50 = "pp_lite_t_seg50"
    PP_LITE_T_SEG75 = "pp_lite_t_seg75"
    PP_LITE_B_SEG = "pp_lite_b_seg"
    PP_LITE_B_SEG50 = "pp_lite_b_seg50"
    PP_LITE_B_SEG75 = "pp_lite_b_seg75"


ARCHITECTURES = {ModelNames.RESNET18: resnet.ResNet18,
                 ModelNames.RESNET34: resnet.ResNet34,
                 ModelNames.RESNET50_3343: resnet.ResNet50_3343,
                 ModelNames.RESNET50: resnet.ResNet50,
                 ModelNames.RESNET101: resnet.ResNet101,
                 ModelNames.RESNET152: resnet.ResNet152,
                 ModelNames.RESNET18_CIFAR: resnet.ResNet18Cifar,
                 ModelNames.CUSTOM_RESNET: resnet.CustomizedResnet,
                 ModelNames.CUSTOM_RESNET50: resnet.CustomizedResnet50,
                 ModelNames.CUSTOM_RESNET_CIFAR: resnet.CustomizedResnetCifar,
                 ModelNames.CUSTOM_RESNET50_CIFAR: resnet.CustomizedResnet50Cifar,
                 ModelNames.MOBILENET_V2: MobileNetV2Base,
                 ModelNames.MOBILE_NET_V2_135: MobileNetV2_135,
                 ModelNames.CUSTOM_MOBILENET_V2: CustomMobileNetV2,
                 ModelNames.MOBILENET_V3_LARGE: mobilenetv3_large,
                 ModelNames.MOBILENET_V3_SMALL: mobilenetv3_small,
                 ModelNames.MOBILENET_V3_CUSTOM: mobilenetv3_custom,
                 ModelNames.CUSTOM_DENSENET: densenet.CustomizedDensnet,
                 ModelNames.DENSENET121: densenet.DenseNet121,
                 ModelNames.DENSENET161: densenet.DenseNet161,
                 ModelNames.DENSENET169: densenet.DenseNet169,
                 ModelNames.DENSENET201: densenet.DenseNet201,
                 ModelNames.SHELFNET18_LW: ShelfNet18_LW,
                 ModelNames.SHELFNET34_LW: ShelfNet34_LW,
                 ModelNames.SHELFNET50_3343: ShelfNet503343,
                 ModelNames.SHELFNET50: ShelfNet50,
                 ModelNames.SHELFNET101: ShelfNet101,
                 ModelNames.SHUFFLENET_V2_X0_5: ShufflenetV2_x0_5,
                 ModelNames.SHUFFLENET_V2_X1_0: ShufflenetV2_x1_0,
                 ModelNames.SHUFFLENET_V2_X1_5: ShufflenetV2_x1_5,
                 ModelNames.SHUFFLENET_V2_X2_0: ShufflenetV2_x2_0,
                 ModelNames.SHUFFLENET_V2_CUSTOM5: CustomizedShuffleNetV2,
                 ModelNames.DARKNET53: Darknet53,
                 ModelNames.CSP_DARKNET53: CSPDarknet53,
                 ModelNames.RESNEXT50: ResNeXt50,
                 ModelNames.RESNEXT101: ResNeXt101,
                 ModelNames.GOOGLENET_V1: GoogleNetV1,
                 ModelNames.EFFICIENTNET_B0: efficientnet.EfficientNetB0,
                 ModelNames.EFFICIENTNET_B1: efficientnet.EfficientNetB1,
                 ModelNames.EFFICIENTNET_B2: efficientnet.EfficientNetB2,
                 ModelNames.EFFICIENTNET_B3: efficientnet.EfficientNetB3,
                 ModelNames.EFFICIENTNET_B4: efficientnet.EfficientNetB4,
                 ModelNames.EFFICIENTNET_B5: efficientnet.EfficientNetB5,
                 ModelNames.EFFICIENTNET_B6: efficientnet.EfficientNetB6,
                 ModelNames.EFFICIENTNET_B7: efficientnet.EfficientNetB7,
                 ModelNames.EFFICIENTNET_B8: efficientnet.EfficientNetB8,
                 ModelNames.EFFICIENTNET_L2: efficientnet.EfficientNetL2,
                 ModelNames.CUSTOMIZEDEFFICIENTNET: efficientnet.CustomizedEfficientnet,
                 ModelNames.REGNETY200: regnet.RegNetY200,
                 ModelNames.REGNETY400: regnet.RegNetY400,
                 ModelNames.REGNETY600: regnet.RegNetY600,
                 ModelNames.REGNETY800: regnet.RegNetY800,
                 ModelNames.CUSTOM_REGNET: regnet.CustomRegNet,
                 ModelNames.NAS_REGNET: regnet.NASRegNet,
                 ModelNames.YOLOX_N: YoloX_N,
                 ModelNames.YOLOX_T: YoloX_T,
                 ModelNames.YOLOX_S: YoloX_S,
                 ModelNames.YOLOX_M: YoloX_M,
                 ModelNames.YOLOX_L: YoloX_L,
                 ModelNames.YOLOX_X: YoloX_X,
                 ModelNames.CUSTOM_YOLO_X: CustomYoloX,
                 ModelNames.SSD_MOBILENET_V1: SSDMobileNetV1,
                 ModelNames.SSD_LITE_MOBILENET_V2: SSDLiteMobileNetV2,
                 ModelNames.REPVGG_A0: repvgg.RepVggA0,
                 ModelNames.REPVGG_A1: repvgg.RepVggA1,
                 ModelNames.REPVGG_A2: repvgg.RepVggA2,
                 ModelNames.REPVGG_B0: repvgg.RepVggB0,
                 ModelNames.REPVGG_B1: repvgg.RepVggB1,
                 ModelNames.REPVGG_B2: repvgg.RepVggB2,
                 ModelNames.REPVGG_B3: repvgg.RepVggB3,
                 ModelNames.REPVGG_D2SE: repvgg.RepVggD2SE,
                 ModelNames.REPVGG_CUSTOM: repvgg.RepVggCustom,
                 ModelNames.DDRNET_23: DDRNet23,
                 ModelNames.DDRNET_23_SLIM: DDRNet23Slim,
                 ModelNames.CUSTOM_DDRNET_23: AnyBackBoneDDRNet23,
                 ModelNames.STDC1_CLASSIFICATION: STDC1Classification,
                 ModelNames.STDC2_CLASSIFICATION: STDC2Classification,
                 ModelNames.STDC1_SEG: STDC1Seg,
                 ModelNames.STDC1_SEG50: STDC1Seg,
                 ModelNames.STDC1_SEG75: STDC1Seg,
                 ModelNames.STDC2_SEG: STDC2Seg,
                 ModelNames.STDC2_SEG50: STDC2Seg,
                 ModelNames.STDC2_SEG75: STDC2Seg,
                 ModelNames.REGSEG48: RegSeg48,
                 ModelNames.KD_MODULE: KDModule,
                 ModelNames.VIT_BASE: ViTBase,
                 ModelNames.VIT_LARGE: ViTLarge,
                 ModelNames.VIT_HUGE: ViTHuge,
                 ModelNames.BEIT_BASE_PATCH16_224: BeitBasePatch16_224,
                 ModelNames.BEIT_LARGE_PATCH16_224: BeitLargePatch16_224,
                 ModelNames.PP_LITE_T_SEG: PPLiteSegT,
                 ModelNames.PP_LITE_T_SEG50: PPLiteSegT,
                 ModelNames.PP_LITE_T_SEG75: PPLiteSegT,
                 ModelNames.PP_LITE_B_SEG: PPLiteSegB,
                 ModelNames.PP_LITE_B_SEG50: PPLiteSegB,
                 ModelNames.PP_LITE_B_SEG75: PPLiteSegB,
                 }

KD_ARCHITECTURES = {
    ModelNames.KD_MODULE: KDModule
}
