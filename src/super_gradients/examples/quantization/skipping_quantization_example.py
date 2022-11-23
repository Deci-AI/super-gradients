import torch
from torch import nn

from super_gradients.training.utils.quantization.core import SkipQuantization
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer


def skipping_quantization_example():
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.conv2 = SkipQuantization(nn.Conv2d(8, 8, kernel_size=3, padding=1))  # can use the wrapper to skip

        def forward(self, x):
            return self.conv2(self.conv1(x))

    module = MyModel()

    # Initialize the quantization utility, register layers to skip, and quantize the module
    q_util = SelectiveQuantizer()
    q_util.register_skip_quantization(layer_names={"conv1"})  # can also configure skip by layer names
    q_util.quantize_module(module)

    print(module)  # You should expect to see Conv2d

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, 8, 32, 32))


if __name__ == "__main__":
    skipping_quantization_example()
