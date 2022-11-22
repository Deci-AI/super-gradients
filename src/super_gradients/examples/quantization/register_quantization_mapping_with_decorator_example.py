import torch
from pytorch_quantization import nn as quant_nn
from torch import nn

from super_gradients.training.utils.quantization.core import SGQuantMixin
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer, register_quantized_module


def register_quantization_mapping_with_decorator_example():
    # ARRANGE
    class MyBlock(nn.Module):
        def __init__(self, in_feats, out_feats) -> None:
            super().__init__()
            self.in_feats = in_feats
            self.out_feats = out_feats
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, x):
            return self.linear(self.flatten(x))

    @register_quantized_module(float_source=MyBlock)
    class MyQuantizedBlock(SGQuantMixin):
        def __init__(self, in_feats, out_feats) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear = quant_nn.QuantLinear(in_feats, out_feats)

        def forward(self, x):
            return self.linear(self.flatten(x))

    class MyModel(nn.Module):
        def __init__(self, res, n_classes) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.my_block = MyBlock(4 * (res**2), n_classes)

        def forward(self, x):
            y = self.conv(x)
            return self.my_block(y)

    res = 32
    n_clss = 10
    module = MyModel(res, n_clss)

    # TEST
    q_util = SelectiveQuantizer()
    q_util.quantize_module(module)

    x = torch.rand(1, 3, res, res)

    print(module)

    # ASSERT
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, n_clss))


if __name__ == "__main__":
    register_quantization_mapping_with_decorator_example()
