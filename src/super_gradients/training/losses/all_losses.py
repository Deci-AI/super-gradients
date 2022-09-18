from torch import nn

from super_gradients.training.losses import LabelSmoothingCrossEntropyLoss, YoLoV3DetectionLoss, ShelfNetOHEMLoss, \
    ShelfNetSemanticEncodingLoss, RSquaredLoss, YoLoV5DetectionLoss, SSDLoss, BCEDiceLoss
from super_gradients.training.losses.stdc_loss import STDCLoss

LOSSES = {"cross_entropy": LabelSmoothingCrossEntropyLoss,
          "mse": nn.MSELoss,
          "r_squared_loss": RSquaredLoss,
          "detection_loss": YoLoV3DetectionLoss,
          "shelfnet_ohem_loss": ShelfNetOHEMLoss,
          "shelfnet_se_loss": ShelfNetSemanticEncodingLoss,
          "yolo_v5_loss": YoLoV5DetectionLoss,
          "ssd_loss": SSDLoss,
          "stdc_loss": STDCLoss,
          "bce_dice_loss": BCEDiceLoss
          }
