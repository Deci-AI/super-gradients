from torch import nn

from super_gradients.training.losses import LabelSmoothingCrossEntropyLoss, ShelfNetOHEMLoss, \
    ShelfNetSemanticEncodingLoss, RSquaredLoss, SSDLoss, BCEDiceLoss, YoloXDetectionLoss, KDLogitsLoss, DiceCEEdgeLoss
from super_gradients.training.losses.stdc_loss import STDCLoss


class LossNames:
    """Static class holding all the supported loss names"""""
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    R_SQUARED_LOSS = "r_squared_loss"
    SHELFNET_OHEM_LOSS = "shelfnet_ohem_loss"
    SHELFNET_SE_LOSS = "shelfnet_se_loss"
    YOLOX_LOSS = "yolox_loss"
    SSD_LOSS = "ssd_loss"
    STDC_LOSS = "stdc_loss"
    BCE_DICE_LOSS = "bce_dice_loss"
    KD_LOSS = "kd_loss"
    DICE_CE_EDGE_LOSS = "dice_ce_edge_loss"


LOSSES = {LossNames.CROSS_ENTROPY: LabelSmoothingCrossEntropyLoss,
          LossNames.MSE: nn.MSELoss,
          LossNames.R_SQUARED_LOSS: RSquaredLoss,
          LossNames.SHELFNET_OHEM_LOSS: ShelfNetOHEMLoss,
          LossNames.SHELFNET_SE_LOSS: ShelfNetSemanticEncodingLoss,
          LossNames.YOLOX_LOSS: YoloXDetectionLoss,
          LossNames.SSD_LOSS: SSDLoss,
          LossNames.STDC_LOSS: STDCLoss,
          LossNames.BCE_DICE_LOSS: BCEDiceLoss,
          LossNames.KD_LOSS: KDLogitsLoss,
          LossNames.DICE_CE_EDGE_LOSS: DiceCEEdgeLoss,
          }
