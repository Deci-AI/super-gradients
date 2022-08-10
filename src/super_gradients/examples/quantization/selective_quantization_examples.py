import torch
from torch import nn

from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
from super_gradients.training.datasets import Cifar10DatasetInterface
from super_gradients.training.utils.quantization.core import SkipQuantization, SGQuantMixin
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer
from pytorch_quantization import nn as quant_nn


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


def non_default_calibrators_example():
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv1(x)

    module = MyModel()

    # Initialize the quantization utility, with different calibrators, and quantize the module
    q_util = SelectiveQuantizer(default_quant_modules_calib_method='max', default_per_channel_quant_modules=True)
    q_util.quantize_module(module)

    print(module)  # You should expect to see QuantConv2d, with different calibrators

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, 8, 32, 32))


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
    q_util.register_skip_quantization(layer_names={'conv1'})  # can also configure skip by layer names
    q_util.quantize_module(module)

    print(module)  # You should expect to see Conv2d, with different calibrators

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, 8, 32, 32))


def dynamic_quantized_mapping():
    class MyBlock(nn.Module):
        def __init__(self, in_feats, out_feats) -> None:
            super().__init__()
            self.in_feats = in_feats
            self.out_feats = out_feats
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, x):
            return self.linear(self.flatten(x))

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
            self.my_block = MyBlock(3 * (res ** 2), n_classes)

        def forward(self, x):
            return self.my_block(x)

    res = 32
    n_clss = 10
    module = MyModel(res, n_clss)

    q_util = SelectiveQuantizer()
    q_util.register_quantization_mapping(layer_names={'my_block'}, quantized_type=MyQuantizedBlock)
    q_util.quantize_module(module)

    print(module)  # You should expect to see QuantizedMyBlock, with different calibrators

    x = torch.rand(1, 3, res, res)
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, n_clss))


def e2e_example():
    class MyBlock(nn.Module):
        def __init__(self, in_feats, out_feats) -> None:
            super().__init__()
            self.in_feats = in_feats
            self.out_feats = out_feats
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, x):
            return self.linear(self.flatten(x))

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
            self.my_block = MyBlock(3 * (res ** 2), n_classes)

        def forward(self, x):
            return self.my_block(x)

    res = 32
    n_clss = 10
    module = MyModel(res, n_clss).cuda()

    # QUANTIZE
    q_util = SelectiveQuantizer()
    q_util.register_quantization_mapping(layer_names={'my_block'}, quantized_type=MyQuantizedBlock)
    q_util.quantize_module(module)

    # CALIBRATE
    dataset_interface = Cifar10DatasetInterface(dataset_params={"batch_size": 32})
    dataset_interface.build_data_loaders()
    train_loader = dataset_interface.train_loader
    calib = QuantizationCalibrator()
    calib.calibrate_model(module, method=q_util.default_quant_modules_calib_method, calib_data_loader=train_loader)

    # SANITY
    x = torch.rand(1, 3, res, res, device='cuda')
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, n_clss))

    # EXPORT TO ONNX
    export_quantized_module_to_onnx(module, "my_quantized_model.onnx", input_shape=(1, 3, res, res))


if __name__ == '__main__':
    vanilla_quantize_all_example()
    non_default_calibrators_example()
    skipping_quantization_example()
    e2e_example()
