from enum import Enum


class UpsampleMode(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    SNPE_BILINEAR = "snpe_bilinear"
