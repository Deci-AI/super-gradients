class Losses:
    """Static class holding all the supported loss names"""""
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    R_SQUARED_LOSS = "r_squared_loss"
    SHELFNET_OHEM_LOSS = "shelfnet_ohem_loss"
    SHELFNET_SE_LOSS = "shelfnet_se_loss"
    YOLOX_LOSS = "yolox_loss"
    YOLOX_FAST_LOSS = "yolox_fast_loss"
    SSD_LOSS = "ssd_loss"
    STDC_LOSS = "stdc_loss"
    BCE_DICE_LOSS = "bce_dice_loss"
    KD_LOSS = "kd_loss"
    DICE_CE_EDGE_LOSS = "dice_ce_edge_loss"


class Metrics:
    """Static class holding all the supported metric names"""""
    ACCURACY = 'Accuracy'
    TOP5 = 'Top5'
    DETECTION_METRICS = 'DetectionMetrics'
    IOU = 'IoU'
    BINARY_IOU = "BinaryIOU"
    DICE = "Dice"
    BINARY_DICE = "BinaryDice"
    PIXEL_ACCURACY = 'PixelAccuracy'


class Transforms:
    """Static class holding all the supported transform names"""""
    # From SG
    SegRandomFlip = "SegRandomFlip"
    SegResize = "SegResize"
    SegRescale = "SegRescale"
    SegRandomRescale = "SegRandomRescale"
    SegRandomRotate = "SegRandomRotate"
    SegCropImageAndMask = "SegCropImageAndMask"
    SegRandomGaussianBlur = "SegRandomGaussianBlur"
    SegPadShortToCropSize = "SegPadShortToCropSize"
    SegColorJitter = "SegColorJitter"
    DetectionMosaic = "DetectionMosaic"
    DetectionRandomAffine = "DetectionRandomAffine"
    DetectionMixup = "DetectionMixup"
    DetectionHSV = "DetectionHSV"
    DetectionHorizontalFlip = "DetectionHorizontalFlip"
    DetectionPaddedRescale = "DetectionPaddedRescale"
    DetectionTargetsFormat = "DetectionTargetsFormat"
    DetectionTargetsFormatTransform = "DetectionTargetsFormatTransform"
    RandomResizedCropAndInterpolation = "RandomResizedCropAndInterpolation"
    RandAugmentTransform = "RandAugmentTransform"
    Lighting = "Lighting"
    RandomErase = "RandomErase"

    # From torch
    Compose = "Compose"
    ToTensor = "ToTensor"
    PILToTensor = "PILToTensor"
    ConvertImageDtype = "ConvertImageDtype"
    ToPILImage = "ToPILImage"
    Normalize = "Normalize"
    Resize = "Resize"
    CenterCrop = "CenterCrop"
    Pad = "Pad"
    Lambda = "Lambda"
    RandomApply = "RandomApply"
    RandomChoice = "RandomChoice"
    RandomOrder = "RandomOrder"
    RandomCrop = "RandomCrop"
    RandomHorizontalFlip = "RandomHorizontalFlip"
    RandomVerticalFlip = "RandomVerticalFlip"
    RandomResizedCrop = "RandomResizedCrop"
    FiveCrop = "FiveCrop"
    TenCrop = "TenCrop"
    LinearTransformation = "LinearTransformation"
    ColorJitter = "ColorJitter"
    RandomRotation = "RandomRotation"
    RandomAffine = "RandomAffine"
    Grayscale = "Grayscale"
    RandomGrayscale = "RandomGrayscale"
    RandomPerspective = "RandomPerspective"
    RandomErasing = "RandomErasing"
    GaussianBlur = "GaussianBlur"
    InterpolationMode = "InterpolationMode"
    RandomInvert = "RandomInvert"
    RandomPosterize = "RandomPosterize"
    RandomSolarize = "RandomSolarize"
    RandomAdjustSharpness = "RandomAdjustSharpness"
    RandomAutocontrast = "RandomAutocontrast"
    RandomEqualize = "RandomEqualize"


class Optimizers:
    """Static class holding all the supported optimizer names"""""
    SGD = "SGD"
    ADAM = "Adam"
    RMS_PROP = "RMSprop"
    RMS_PROP_TF = "RMSpropTF"
    LAMB = "Lamb"


class Callbacks:
    """Static class holding all the supported callback names"""""
    DECI_LAB_UPLOAD = 'DeciLabUploadCallback'
    LR_CALLBACK_BASE = 'LRCallbackBase'
    LR_SCHEDULER = 'LRSchedulerCallback'
    METRICS_UPDATE = 'MetricsUpdateCallback'
    MODEL_CONVERSION_CHECK = 'ModelConversionCheckCallback'
    EARLY_STOP = 'EarlyStop'
    DETECTION_MULTISCALE_PREPREDICTION = 'DetectionMultiscalePrePredictionCallback'
    YOLOX_TRAINING_STAGE_SWITCH = 'YoloXTrainingStageSwitchCallback'


class LRSchedulers:
    """Static class to hold all the supported LR Scheduler names"""""
    STEP = "step"
    POLY = "poly"
    COSINE = "cosine"
    EXP = "exp"
    FUNCTION = "function"


class LRWarmups:
    """Static class to hold all the supported LR Warmup names"""""
    LINEAR_STEP = "linear_step"


class Samplers:
    """Static class to hold all the supported Samplers names"""""
    INFINITE = "InfiniteSampler"
    REPEAT_AUG = "RepeatAugSampler"
    DISTRIBUTED = "DistributedSampler"


class ContextModules:
    """Static class to hold all the segmentation context module names"""""
    ASPP = "aspp"
    SPPM = "sppm"


class Models:
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
    CUSTOM_ANYNET = "custom_anynet"
    NAS_REGNET = "nas_regnet"
    YOLOX_N = "yolox_n"
    YOLOX_T = "yolox_t"
    YOLOX_S = "yolox_s"
    YOLOX_M = "yolox_m"
    YOLOX_L = "yolox_l"
    YOLOX_X = "yolox_x"
    CUSTOM_YOLO_X = "custom_yolox"
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
    CUSTOM_STDC = 'custom_stdc'
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
    UNET_CUSTOM = "unet_custom"
