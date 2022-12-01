from torch import nn


class Residual(nn.Module):
    """
    This is a placeholder module used by the quantization engine only.
    This module will be replaced by converters.
    The module is to be used as a residual skip connection within a single block.
    """

    def forward(self, x):
        return x


class SkipConnection(nn.Module):
    """
    This is a placeholder module used by the quantization engine only.
    This module will be replaced by converters.
    The module is to be used as a skip connection between blocks.
    """

    def forward(self, x):
        return x


class BackboneInternalSkipConnection(SkipConnection):
    """
    This is a placeholder module used by the quantization engine only.
    This module will be replaced by converters.
    The module is to be used as a skip connection between blocks inside the backbone.
    """


class HeadInternalSkipConnection(SkipConnection):
    """
    This is a placeholder module used by the quantization engine only.
    This module will be replaced by converters.
    The module is to be used as a skip connection between blocks inside the head.
    """


class CrossModelSkipConnection(SkipConnection):
    """
    This is a placeholder module used by the quantization engine only.
    This module will be replaced by converters.
    The module is to be used as a skip connection between backbone and the head.
    """
