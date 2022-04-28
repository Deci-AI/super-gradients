from torch import nn

from super_gradients.training.losses import LabelSmoothingCrossEntropyLoss, ShelfNetOHEMLoss, \
    ShelfNetSemanticEncodingLoss, RSquaredLoss, SSDLoss, BCEDiceLoss
from super_gradients.training.losses.stdc_loss import STDCLoss

LOSSES = {"cross_entropy": LabelSmoothingCrossEntropyLoss,
          "mse": nn.MSELoss,
          "r_squared_loss": RSquaredLoss,
          "shelfnet_ohem_loss": ShelfNetOHEMLoss,
          "shelfnet_se_loss": ShelfNetSemanticEncodingLoss,
          "ssd_loss": SSDLoss,
          "stdc_loss": STDCLoss,
          "bce_dice_loss": BCEDiceLoss
          }
