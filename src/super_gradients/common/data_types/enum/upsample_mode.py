from enum import Enum


class UpsampleMode(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    SNPE_BILINEAR = "snpe_bilinear"


class DownSampleMode(Enum):
    MAX_POOL = "max_pool"
    ANTI_ALIAS = "anti_alias"
