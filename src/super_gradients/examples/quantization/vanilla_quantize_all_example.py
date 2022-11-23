import torch
from torch import nn

from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer


def vanilla_quantize_all_example():
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv1(x)

    module = MyModel()

    # Initialize the quantization utility, and quantize the module
    q_util = SelectiveQuantizer()
    q_util.quantize_module(module)

    print(module)  # You should expect to see QuantConv2d

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, 8, 32, 32))


if __name__ == "__main__":
    vanilla_quantize_all_example()
