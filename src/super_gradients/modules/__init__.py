from .conv_bn_act_block import ConvBNAct
from .conv_bn_relu_block import ConvBNReLU
from .repvgg_block import RepVGGBlock
from .se_blocks import SEBlock, EffectiveSEBlock
from .skip_connections import Residual, SkipConnection, CrossModelSkipConnection, BackboneInternalSkipConnection, HeadInternalSkipConnection
from super_gradients.common.abstractions.abstract_logger import get_logger

__all__ = [
    "ConvBNAct",
    "ConvBNReLU",
    "RepVGGBlock",
    "SEBlock",
    "EffectiveSEBlock",
    "Residual",
    "SkipConnection",
    "CrossModelSkipConnection",
    "BackboneInternalSkipConnection",
    "HeadInternalSkipConnection",
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

    quant_extensions = [
        "QuantBottleneck",
        "QuantResidual",
        "QuantSkipConnection",
        "QuantCrossModelSkipConnection",
        "QuantBackboneInternalSkipConnection",
        "QuantHeadInternalSkipConnection",
    ]

except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug(f"Failed to import pytorch_quantization: {import_err}")
    quant_extensions = None


if quant_extensions is not None:
    __all__.extend(quant_extensions)
