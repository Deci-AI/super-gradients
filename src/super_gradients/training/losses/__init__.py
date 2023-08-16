from super_gradients.training.losses.focal_loss import FocalLoss
from super_gradients.training.losses.kd_losses import KDLogitsLoss
from super_gradients.training.losses.label_smoothing_cross_entropy_loss import LabelSmoothingCrossEntropyLoss
from super_gradients.training.losses.r_squared_loss import RSquaredLoss
from super_gradients.training.losses.shelfnet_ohem_loss import ShelfNetOHEMLoss
from super_gradients.training.losses.shelfnet_semantic_encoding_loss import ShelfNetSemanticEncodingLoss
from super_gradients.training.losses.yolox_loss import YoloXDetectionLoss, YoloXFastDetectionLoss
from super_gradients.training.losses.ssd_loss import SSDLoss
from super_gradients.training.losses.bce_dice_loss import BCEDiceLoss
from super_gradients.training.losses.dice_ce_edge_loss import DiceCEEdgeLoss
from super_gradients.training.losses.ppyolo_loss import PPYoloELoss
from super_gradients.training.losses.dekr_loss import DEKRLoss
from super_gradients.training.losses.stdc_loss import STDCLoss
from super_gradients.training.losses.rescoring_loss import RescoringLoss

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import LOSSES

__all__ = [
    "LOSSES",
    "Losses",
    "FocalLoss",
    "LabelSmoothingCrossEntropyLoss",
    "ShelfNetOHEMLoss",
    "ShelfNetSemanticEncodingLoss",
    "YoloXDetectionLoss",
    "YoloXFastDetectionLoss",
    "RSquaredLoss",
    "SSDLoss",
    "BCEDiceLoss",
    "KDLogitsLoss",
    "DiceCEEdgeLoss",
    "PPYoloELoss",
    "DEKRLoss",
    "STDCLoss",
    "RescoringLoss",
]
