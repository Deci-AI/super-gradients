from torch import nn


def drop_path(x, drop_prob: float = 0.0, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Intended usage of this block is the following:

    >>> class ResNetBlock(nn.Module):
    >>>   def __init__(self, ..., drop_path_rate:float):
    >>>     self.drop_path = DropPath(drop_path_rate)
    >>>
    >>>   def forward(self, x):
    >>>     return x + self.drop_path(self.conv_bn_act(x))

    Code taken from TIMM (https://github.com/rwightman/pytorch-image-models)
    Apache License 2.0
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """

        :param drop_prob: Probability of zeroing out individual vector (channel dimension) of each feature map
        :param scale_by_keep: Whether to scale the output by the keep probability. Enable by default and helps to
                              keep output mean & std in the same range as w/o drop path.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        return drop_path(x, self.drop_prob, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
