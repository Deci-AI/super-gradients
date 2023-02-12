from super_gradients.training.models.segmentation_models.stdc.stdc_encoder import STDCBackbone
from super_gradients.training.models.segmentation_models.stdc.stdc_decoder import STDCDecoder
from super_gradients.training.models.segmentation_models.unet.unet_encoder import UNetBackbone, UNetEncoder
from super_gradients.training.models.segmentation_models.unet.unet_decoder import UNetDecoder

SEGMENTATION_BACKBONES = dict(
    UNetBackbone=UNetBackbone,
    STDCBackbone=STDCBackbone,
)

SEGMENTATION_ENCODERS = dict(
    UNetEncoder=UNetEncoder,
)

SEGMENTATION_DECODERS = dict(
    UNetDecoder=UNetDecoder,
    STDCDecoder=STDCDecoder,
)
