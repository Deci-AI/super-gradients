from torch import nn
from super_gradients.common.object_names import Losses
from super_gradients.training.losses import (
    LabelSmoothingCrossEntropyLoss,
    ShelfNetOHEMLoss,
    ShelfNetSemanticEncodingLoss,
    RSquaredLoss,
    SSDLoss,
    BCEDiceLoss,
    YoloXDetectionLoss,
    YoloXFastDetectionLoss,
    KDLogitsLoss,
    DiceCEEdgeLoss,
)
from super_gradients.training.losses.stdc_loss import STDCLoss


LOSSES = {
    Losses.CROSS_ENTROPY: LabelSmoothingCrossEntropyLoss,
    Losses.MSE: nn.MSELoss,
    Losses.R_SQUARED_LOSS: RSquaredLoss,
    Losses.SHELFNET_OHEM_LOSS: ShelfNetOHEMLoss,
    Losses.SHELFNET_SE_LOSS: ShelfNetSemanticEncodingLoss,
    Losses.YOLOX_LOSS: YoloXDetectionLoss,
    Losses.YOLOX_FAST_LOSS: YoloXFastDetectionLoss,
    Losses.SSD_LOSS: SSDLoss,
    Losses.STDC_LOSS: STDCLoss,
    Losses.BCE_DICE_LOSS: BCEDiceLoss,
    Losses.KD_LOSS: KDLogitsLoss,
    Losses.DICE_CE_EDGE_LOSS: DiceCEEdgeLoss,
}
