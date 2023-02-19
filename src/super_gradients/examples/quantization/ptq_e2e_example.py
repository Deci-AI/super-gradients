import torch
from pytorch_quantization import nn as quant_nn
from torch import nn

from super_gradients.training.dataloaders import cifar10_train
from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
from super_gradients.training.utils.quantization.core import SGQuantMixin
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer


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
            self.my_block = MyBlock(3 * (res**2), n_classes)

        def forward(self, x):
            return self.my_block(x)

    res = 32
    n_clss = 10
    module = MyModel(res, n_clss)

    # QUANTIZE
    q_util = SelectiveQuantizer()
    q_util.register_quantization_mapping(layer_names={"my_block"}, quantized_target_class=MyQuantizedBlock)
    q_util.quantize_module(module)

    # CALIBRATE (PTQ)
    train_loader = cifar10_train()
    calib = QuantizationCalibrator()
    calib.calibrate_model(module, method=q_util.default_quant_modules_calibrator_inputs, calib_data_loader=train_loader)

    module.cuda()
    # SANITY
    x = torch.rand(1, 3, res, res, device="cuda")
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, n_clss))

    print(module)

    # EXPORT TO ONNX
    export_quantized_module_to_onnx(module, "my_quantized_model.onnx", input_shape=(1, 3, res, res))


if __name__ == "__main__":
    e2e_example()
