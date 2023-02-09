import torch
from torch import nn

from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer


def non_default_calibrators_example():
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv1(x)

    module = MyModel()

    # Initialize the quantization utility, with different calibrators, and quantize the module
    q_util = SelectiveQuantizer(
        default_quant_modules_calibrator_weights="percentile",
        default_quant_modules_calibrator_inputs="entropy",
        default_per_channel_quant_weights=False,
        default_learn_amax=False,
    )
    q_util.quantize_module(module)

    print(module)  # You should expect to see QuantConv2d, with Histogram calibrators

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, 8, 32, 32))


if __name__ == "__main__":
    non_default_calibrators_example()
