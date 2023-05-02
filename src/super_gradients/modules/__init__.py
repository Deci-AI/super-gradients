from super_gradients.modules.anti_alias import AntiAliasDownsample
from super_gradients.modules.pixel_shuffle import PixelShuffle
from super_gradients.modules.pose_estimation_modules import LightweightDEKRHead
from super_gradients.modules.conv_bn_act_block import ConvBNAct, Conv
from super_gradients.modules.conv_bn_relu_block import ConvBNReLU
from super_gradients.modules.repvgg_block import RepVGGBlock
from super_gradients.modules.qarepvgg_block import QARepVGGBlock
from super_gradients.modules.se_blocks import SEBlock, EffectiveSEBlock
from super_gradients.modules.skip_connections import (
    Residual,
    SkipConnection,
    CrossModelSkipConnection,
    BackboneInternalSkipConnection,
    HeadInternalSkipConnection,
)
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import ALL_DETECTION_MODULES

from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.modules.detection_modules import (
    PANNeck,
    NHeads,
    MultiOutputBackbone,
    NStageBackbone,
    MobileNetV1Backbone,
    MobileNetV2Backbone,
    SSDNeck,
    SSDInvertedResidualNeck,
    SSDBottleneckNeck,
    SSDHead,
)
from super_gradients.module_interfaces import SupportsReplaceNumClasses

__all__ = [
    "BaseDetectionModule",
    "ALL_DETECTION_MODULES",
    "PixelShuffle",
    "AntiAliasDownsample",
    "Conv",
    "ConvBNAct",
    "ConvBNReLU",
    "RepVGGBlock",
    "QARepVGGBlock",
    "SEBlock",
    "EffectiveSEBlock",
    "Residual",
    "SkipConnection",
    "CrossModelSkipConnection",
    "BackboneInternalSkipConnection",
    "HeadInternalSkipConnection",
    "LightweightDEKRHead",
    "PANNeck",
    "NHeads",
    "MultiOutputBackbone",
    "NStageBackbone",
    "MobileNetV1Backbone",
    "MobileNetV2Backbone",
    "SSDNeck",
    "SSDInvertedResidualNeck",
    "SSDBottleneckNeck",
    "SSDHead",
    "SupportsReplaceNumClasses",
]

logger = get_logger(__name__)
try:
    # flake8 respects only the first occurence of __all__ defined in the module's root
    from .quantization import QuantBottleneck  # noqa: F401

    from .quantization import QuantResidual  # noqa: F401
    from .quantization import QuantSkipConnection  # noqa: F401
    from .quantization import QuantCrossModelSkipConnection  # noqa: F401
    from .quantization import QuantBackboneInternalSkipConnection  # noqa: F401
    from .quantization import QuantHeadInternalSkipConnection  # noqa: F401

    from .quantization import QuantSTDCBlock  # noqa: F401
    from .quantization import QuantAttentionRefinementModule  # noqa: F401
    from .quantization import QuantFeatureFusionModule  # noqa: F401
    from .quantization import QuantContextPath  # noqa: F401

    quant_extensions = [
        "QuantBottleneck",
        "QuantResidual",
        "QuantSkipConnection",
        "QuantCrossModelSkipConnection",
        "QuantBackboneInternalSkipConnection",
        "QuantHeadInternalSkipConnection",
        "QuantSTDCBlock",
        "QuantAttentionRefinementModule",
        "QuantFeatureFusionModule",
        "QuantContextPath",
    ]

except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug(f"Failed to import pytorch_quantization: {import_err}")
    quant_extensions = None


if quant_extensions is not None:
    __all__.extend(quant_extensions)
