from torch import nn

from super_gradients.training.losses import LabelSmoothingCrossEntropyLoss, ShelfNetOHEMLoss, \
    ShelfNetSemanticEncodingLoss, RSquaredLoss, SSDLoss, BCEDiceLoss, YoloXDetectionLoss, KDLogitsLoss, DiceCEEdgeLoss
from super_gradients.training.losses.stdc_loss import STDCLoss


class LossNames:
    """Static class to group the names of every implemented loss"""
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
#
# #######################################################
# from dataclasses import dataclass
#
#
# class SupportedStringsWrapper(object):
#     def __init__(self, string_cls: object):
#         self.string_cls = string_cls
#
#     def get(self, name: str) -> str:
#         """Dynamic wrapper to get access to an attribute."""
#         if hasattr(self.string_cls, name):
#             return getattr(self.string_cls, name)
#         else:
#             raise ValueError(f"{name} is not an implemented in {self.string_cls}")
#
#     def get_all(self):
#         return list(self.string_cls.__dict__.values())
#
#
# @SupportedStringsWrapper
# class LossNames:
#     """Static class to group the names of every implemented loss"""
#     CROSS_ENTROPY = "cross_entropy"
#     MSE = "mse"
#     R_SQUARED_LOSS = "r_squared_loss"
#     SHELFNET_OHEM_LOSS = "shelfnet_ohem_loss"
#     SHELFNET_SE_LOSS = "shelfnet_se_loss"
#     YOLOX_LOSS = "yolox_loss"
#     SSD_LOSS = "ssd_loss"
#     STDC_LOSS = "stdc_loss"
#     BCE_DICE_LOSS = "bce_dice_loss"
#     KD_LOSS = "kd_loss"
#     DICE_CE_EDGE_LOSS = "dice_ce_edge_loss"
#
# all_losses = LossNames()
#
#
# LOSSES = {LossNames.CROSS_ENTROPY: LabelSmoothingCrossEntropyLoss,
#           LossNames.MSE: nn.MSELoss,
#           LossNames.R_SQUARED_LOSS: RSquaredLoss,
#           LossNames.SHELFNET_OHEM_LOSS: ShelfNetOHEMLoss,
#           LossNames.SHELFNET_SE_LOSS: ShelfNetSemanticEncodingLoss,
#           LossNames.YOLOX_LOSS: YoloXDetectionLoss,
#           LossNames.SSD_LOSS: SSDLoss,
#           LossNames.STDC_LOSS: STDCLoss,
#           LossNames.BCE_DICE_LOSS: BCEDiceLoss,
#           LossNames.KD_LOSS: KDLogitsLoss,
#           LossNames.DICE_CE_EDGE_LOSS: DiceCEEdgeLoss,
#           }
# ##################################
# # Not very good imo because the user might just want to write a string...
#
# class LossNames:
#     CROSS_ENTROPY = LabelSmoothingCrossEntropyLoss
#     MSE = nn.MSELoss
#     R_SQUARED_LOSS = RSquaredLoss
#     SHELFNET_OHEM_LOSS = ShelfNetOHEMLoss
#     SHELFNET_SE_LOSS = ShelfNetSemanticEncodingLoss
#     YOLOX_LOSS = YoloXDetectionLoss
#     SSD_LOSS = SSDLoss
#     STDC_LOSS = STDCLoss
#     BCE_DICE_LOSS = BCEDiceLoss
#     KD_LOSS = KDLogitsLoss
#     DICE_CE_EDGE_LOSS = DiceCEEdgeLoss
#
# #################################
# from enum import Enum
#
#
# class LossNamesEnum(Enum, str):
#     """Static class to group the names of every implemented loss"""
#     CROSS_ENTROPY = "cross_entropy"
#     MSE = "mse"
#     R_SQUARED_LOSS = "r_squared_loss"
#     SHELFNET_OHEM_LOSS = "shelfnet_ohem_loss"
#     SHELFNET_SE_LOSS = "shelfnet_se_loss"
#     YOLOX_LOSS = "yolox_loss"
#     SSD_LOSS = "ssd_loss"
#     STDC_LOSS = "stdc_loss"
#     BCE_DICE_LOSS = "bce_dice_loss"
#     KD_LOSS = "kd_loss"
#     DICE_CE_EDGE_LOSS = "dice_ce_edge_loss"
#
#     @staticmethod
#     def get(name: str) -> str:
#         """Dynamic wrapper to get access to an attribute."""
#         if hasattr(LossNames, name):
#             return getattr(LossNames, name)
#         else:
#             raise ValueError(f"{name} is not an implemented LossNames")
#
#     def build_loss(self):
#         return LOSSES[self.value]
