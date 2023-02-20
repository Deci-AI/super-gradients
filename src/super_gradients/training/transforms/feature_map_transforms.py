from torch import nn


class FeatureMapTransforms:
    Upsample = "Upsample"
    ConvTranspose = "ConvTranspose"
    PixelShuffle = "PixelShuffle"


FEATURE_MAP_TRANSFORMS = {
    FeatureMapTransforms.Upsample: nn.Upsample,
    FeatureMapTransforms.ConvTranspose: nn.ConvTranspose2d,
    FeatureMapTransforms.PixelShuffle: nn.PixelShuffle,
}
