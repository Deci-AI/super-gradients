from super_gradients.training.models.segmentation_models.stdc import STDCBackbone
from super_gradients.training.models.segmentation_models.unet.unet_encoder import UNetBackboneBase

SEGMENTATION_BACKBONES = dict(
    UNET=UNetBackboneBase,
    STDC=STDCBackbone,
)
