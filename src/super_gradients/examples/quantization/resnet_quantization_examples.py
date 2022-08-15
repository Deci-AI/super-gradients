from collections import OrderedDict

import torch

from pytorch_quantization import nn as quant_nn
from torch import nn

from super_gradients.training.datasets import ImageNetDatasetInterface
from super_gradients.training.models import Bottleneck, ResNet50
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
from super_gradients.training.utils.quantization.core import SGQuantMixin, QuantizedMetadata
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer


def _calibrate_and_export(module, method, onnx_name):
    module = module.cuda()
    # CALIBRATE (PTQ)
    dataset_interface = ImageNetDatasetInterface({
        "batch_size": 16,
        "val_batch_size": 16,
    })
    dataset_interface.build_data_loaders()
    train_loader = dataset_interface.train_loader
    calib = QuantizationCalibrator()
    calib.calibrate_model(module, method=method, calib_data_loader=train_loader)

    # SANITY
    res = 320
    input_shape = (1, 3, res, res)
    x = torch.rand(*input_shape, device='cuda')
    with torch.no_grad():
        y = module(x)
        torch.testing.assert_close(y.size(), (1, 1000))

    # EXPORT TO ONNX
    export_quantized_module_to_onnx(module, f"{onnx_name}.onnx", input_shape=input_shape)


class QuantBottleneck1(SGQuantMixin):
    """
    Wrap the float module, insert quantized tensor to the shortcut (=residual) layer, so that it would be quantized
    NOTE: we must quantize the float instance, so the mode should be
          QuantizedMetadata.ReplacementAction.QUANTIZE_CHILDREN_THEN_REPLACE
    ANOTHER OPTION IS TO REIMPLEMENT `forward` AND HOLD `residual_quantizer` IN THIS CLASS
    """

    def __init__(self, bottleneck: Bottleneck) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.bottleneck.shortcut.add_module("residual_quantizer",
                                            quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input))

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(OrderedDict({f'bottleneck.{k}': v for k, v in state_dict.items()}), strict)

    @classmethod
    def from_float(cls, float_instance, **kwargs):
        return cls(float_instance)

    def forward(self, x):
        return self.bottleneck.forward(x)


class QuantBottleneck2(SGQuantMixin):
    """
    This is a "hacky" way to achieve proper quantization. Instead of wrapping the float module,
    we just insert quantized tensor to the shortcut (=residual) layer, so that it would be quantized
    NOTE: we must quantize the float instance, so the mode should be
          QuantizedMetadata.ReplacementAction.QUANTIZE_CHILDREN_THEN_REPLACE
    """

    @classmethod
    def from_float(cls, float_instance: Bottleneck, **kwargs):
        float_instance.shortcut.add_module("residual_quantizer",
                                           quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input))
        return float_instance


def sg_resnet50_custom_quantization():

    model = ResNet50(HpmStruct(**{'num_classes': 1000}))

    quantized_type = QuantBottleneck1
    # quantized_type = QuantBottleneck2  # CAN ALSO USE QuantBottleneck2

    mappings = {
        Bottleneck: QuantizedMetadata(float_source=Bottleneck, quantized_type=quantized_type,
                                      action=QuantizedMetadata.ReplacementAction.QUANTIZE_CHILDREN_THEN_REPLACE),
        nn.AdaptiveAvgPool2d: QuantizedMetadata(float_source=nn.AdaptiveAvgPool2d,
                                                quantized_type=None,
                                                action=QuantizedMetadata.ReplacementAction.SKIP)
    }
    sq_util = SelectiveQuantizer(custom_mappings=mappings, default_quant_modules_calib_method='max',
                                 default_per_channel_quant_modules=True)
    sq_util.quantize_module(model, preserve_state_dict=True)
    print(model)
    _calibrate_and_export(model, method='max', onnx_name='sg_resnet50_qdq')


if __name__ == '__main__':
    sg_resnet50_custom_quantization()
