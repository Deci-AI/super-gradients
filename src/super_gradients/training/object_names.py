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
    DETECTION_METRICS_050_095 = 'DetectionMetrics_050_095'
    DETECTION_METRICS_050 = 'DetectionMetrics_050'
    DETECTION_METRICS_075 = 'DetectionMetrics_075'
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
