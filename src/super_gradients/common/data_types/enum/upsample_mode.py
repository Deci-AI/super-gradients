from enum import Enum


class UpsampleMode(Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    NN_PIXEL_SHUFFLE = "nn_pixel_shuffle"
    PIXEL_SHUFFLE = "pixel_shuffle"
