# PACKAGE IMPORTS FOR EXTERNAL USAGE
import cv2

from super_gradients.training.transforms.transforms import (
    DetectionStandardize,
    DetectionMosaic,
    DetectionRandomAffine,
    DetectionHSV,
    DetectionRGB2BGR,
    DetectionPaddedRescale,
    DetectionTargetsFormatTransform,
    Standardize,
    DetectionTransform,
    OpticalFlowColorJitter,
    OpticalFlowOcclusion,
    OpticalFlowRandomRescale,
    OpticalFlowRandomFlip,
    OpticalFlowCrop,
    OpticalFlowInputPadder,
    OpticalFlowNormalize,
)
from super_gradients.training.transforms.keypoints import (
    AbstractKeypointTransform,
    KeypointTransform,
    KeypointsHSV,
    KeypointsRescale,
    KeypointsCompose,
    KeypointsMosaic,
    KeypointsMixup,
    KeypointsRandomAffineTransform,
    KeypointsRandomRotate90,
    KeypointsRandomVerticalFlip,
    KeypointsRandomHorizontalFlip,
    KeypointsImageToTensor,
    KeypointsImageNormalize,
    KeypointsImageStandardize,
    KeypointsPadIfNeeded,
    KeypointsBrightnessContrast,
    KeypointsLongestMaxSize,
    KeypointsReverseImageChannels,
    KeypointsRemoveSmallObjects,
)
from super_gradients.common.object_names import Transforms
from super_gradients.common.registry.registry import TRANSFORMS
from super_gradients.common.registry.albumentation import ALBUMENTATIONS_TRANSFORMS, ALBUMENTATIONS_COMP_TRANSFORMS, imported_albumentations_failure
from super_gradients.training.transforms.detection import AbstractDetectionTransform, DetectionPadIfNeeded, DetectionLongestMaxSize
from super_gradients.training.transforms.optical_flow import AbstractOpticalFlowTransform

__all__ = [
    "TRANSFORMS",
    "ALBUMENTATIONS_TRANSFORMS",
    "ALBUMENTATIONS_COMP_TRANSFORMS",
    "imported_albumentations_failure",
    "Transforms",
    "DetectionTransform",
    "AbstractDetectionTransform",
    "DetectionStandardize",
    "DetectionMosaic",
    "DetectionRandomAffine",
    "DetectionHSV",
    "DetectionRGB2BGR",
    "DetectionPaddedRescale",
    "DetectionTargetsFormatTransform",
    "Standardize",
    "AbstractKeypointTransform",
    "KeypointTransform",
    "KeypointsBrightnessContrast",
    "KeypointsCompose",
    "KeypointsHSV",
    "KeypointsImageNormalize",
    "KeypointsImageStandardize",
    "KeypointsLongestMaxSize",
    "KeypointsMixup",
    "KeypointsMosaic",
    "KeypointsPadIfNeeded",
    "KeypointsRandomAffineTransform",
    "KeypointsRandomHorizontalFlip",
    "KeypointsRandomVerticalFlip",
    "KeypointsRescale",
    "KeypointsRandomRotate90",
    "KeypointsImageToTensor",
    "KeypointsRemoveSmallObjects",
    "KeypointsReverseImageChannels",
    "DetectionPadIfNeeded",
    "DetectionLongestMaxSize",
    "AbstractDetectionTransform",
    "AbstractOpticalFlowTransform",
    "OpticalFlowColorJitter",
    "OpticalFlowOcclusion",
    "OpticalFlowRandomRescale",
    "OpticalFlowRandomFlip",
    "OpticalFlowCrop",
    "OpticalFlowInputPadder",
    "OpticalFlowNormalize",
]

cv2.setNumThreads(0)
