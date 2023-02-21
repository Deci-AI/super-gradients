import unittest
import torch
import torchvision
from torch import nn

from super_gradients.common.object_names import Models

try:
    import super_gradients
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer, register_quantized_module
    from pytorch_quantization.calib import MaxCalibrator, HistogramCalibrator
    from super_gradients.training.utils.quantization.core import SkipQuantization, SGQuantMixin, QuantizedMapping, QuantizedMetadata
    from pytorch_quantization.nn import QuantConv2d
    from pytorch_quantization.tensor_quant import QuantDescriptor

    _imported_pytorch_quantization_failure = False

except (ImportError, NameError, ModuleNotFoundError):
    _imported_pytorch_quantization_failure = True


@unittest.skipIf(_imported_pytorch_quantization_failure, "Failed to import `pytorch_quantization`")
class QuantizationUtilityTest(unittest.TestCase):
    def test_vanilla_replacement(self):
        # ARRANGE
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv1(x)

        module = MyModel()

        # TEST
        q_util = SelectiveQuantizer()
        q_util.quantize_module(module)

        x = torch.rand(1, 3, 32, 32)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, 8, 32, 32))

        self.assertTrue(isinstance(module.conv1, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))

    def test_module_list_replacement(self):
        # ARRANGE
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.convs = nn.ModuleList([nn.Conv2d(3, 8, kernel_size=3, padding=1) for _ in range(3)])

            def forward(self, x):
                return torch.cat([conv(x) for conv in self.convs], dim=1)

        module = MyModel()

        # TEST
        q_util = SelectiveQuantizer()
        q_util.quantize_module(module)

        x = torch.rand(1, 3, 32, 32)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, 3 * 8, 32, 32))

        for conv in module.convs:
            self.assertTrue(isinstance(conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))

    def test_sequential_list_replacement(self):
        # ARRANGE
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.convs = nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=3, padding=1),
                    nn.Conv2d(8, 16, kernel_size=3, padding=1),
                )

            def forward(self, x):
                return self.convs(x)

        module = MyModel()

        # TEST
        q_util = SelectiveQuantizer()
        q_util.quantize_module(module)

        x = torch.rand(1, 3, 32, 32)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, 16, 32, 32))

        for conv in module.convs:
            self.assertTrue(isinstance(conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))

    def test_nested_module_replacement(self):
        # ARRANGE
        class MyBlock(nn.Module):
            def __init__(self, in_feats, out_feats) -> None:
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(in_feats, out_feats)

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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block.linear, SelectiveQuantizer.mapping_instructions[nn.Linear].quantized_target_class))

    def test_static_selective_skip_quantization(self):
        # ARRANGE
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.conv2 = SkipQuantization(nn.Conv2d(8, 16, kernel_size=3, padding=1))

            def forward(self, x):
                return self.conv2(self.conv1(x))

        module = MyModel()

        # TEST
        q_util = SelectiveQuantizer()
        q_util.quantize_module(module)

        x = torch.rand(1, 3, 32, 32)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, 16, 32, 32))

        self.assertTrue(isinstance(module.conv1, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.conv2, nn.Conv2d))

    def test_dynamic_skip_quantization(self):
        # ARRANGE
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv2(self.conv1(x))

        module = MyModel()

        # TEST
        q_util = SelectiveQuantizer()
        q_util.register_skip_quantization(layer_names={"conv2"})
        q_util.quantize_module(module)
        x = torch.rand(1, 3, 32, 32)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, 16, 32, 32))

        self.assertTrue(isinstance(module.conv1, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.conv2, nn.Conv2d))

    def test_custom_quantized_mapping_wrapper_explicit_from_float(self):
        # ARRANGE
        class MyBlock(nn.Module):
            def __init__(self, in_feats, out_feats) -> None:
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(in_feats, out_feats)

            def forward(self, x):
                return self.linear(self.flatten(x))

        class MyQuantizedBlock(SGQuantMixin):
            # NOTE: **kwargs are necessary because quant descriptors are passed there!
            @classmethod
            def from_float(cls, float_instance: MyBlock, **kwargs):
                return cls(in_feats=float_instance.linear.in_features, out_feats=float_instance.linear.out_features)

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
                self.my_block = QuantizedMapping(float_module=MyBlock(4 * (res**2), n_classes), quantized_target_class=MyQuantizedBlock)

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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))

    def test_custom_quantized_mapping_wrapper_implicit_from_float(self):
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
                self.my_block = QuantizedMapping(float_module=MyBlock(4 * (res**2), n_classes), quantized_target_class=MyQuantizedBlock)

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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))

    def test_custom_quantized_mapping_register_with_decorator(self):
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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))
        self.assertTrue(MyQuantizedBlock is not None)
        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))

    def test_dynamic_quantized_mapping(self):
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
        q_util.register_quantization_mapping(layer_names={"my_block"}, quantized_target_class=MyQuantizedBlock)
        q_util.quantize_module(module)

        x = torch.rand(1, 3, res, res)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))

    def test_non_default_quant_descriptors_are_piped(self):
        # ARRANGE
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv1(x)

        module = MyModel()

        # TEST
        q_util = SelectiveQuantizer(default_quant_modules_calibrator_inputs="max", default_quant_modules_calibrator_weights="max")
        q_util.quantize_module(module)

        x = torch.rand(1, 3, 32, 32)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, 8, 32, 32))
        self.assertTrue(isinstance(module.conv1, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(type(module.conv1._input_quantizer._calibrator) == MaxCalibrator)
        self.assertTrue(type(module.conv1._weight_quantizer._calibrator) == MaxCalibrator)

    def test_different_quant_descriptors_are_piped(self):
        # ARRANGE
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv2(self.conv1(x))

        module = MyModel()

        # TEST
        q_util = SelectiveQuantizer()
        q_util.register_quantization_mapping(
            layer_names={"conv1"},
            quantized_target_class=QuantConv2d,
            input_quant_descriptor=QuantDescriptor(calib_method="max"),
            weights_quant_descriptor=QuantDescriptor(calib_method="histogram"),
        )
        q_util.register_quantization_mapping(
            layer_names={"conv2"},
            quantized_target_class=QuantConv2d,
            input_quant_descriptor=QuantDescriptor(calib_method="histogram"),
            weights_quant_descriptor=QuantDescriptor(calib_method="max"),
        )
        q_util.quantize_module(module)

        x = torch.rand(1, 3, 32, 32)

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, 8, 32, 32))
        self.assertTrue(isinstance(module.conv1, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(type(module.conv1._input_quantizer._calibrator) == MaxCalibrator)
        self.assertTrue(type(module.conv1._weight_quantizer._calibrator) == HistogramCalibrator)
        self.assertTrue(type(module.conv2._input_quantizer._calibrator) == HistogramCalibrator)
        self.assertTrue(type(module.conv2._weight_quantizer._calibrator) == MaxCalibrator)

    def test_quant_descriptors_are_piped_to_custom_quant_modules_if_has_kwargs(self):
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

        class MyQuantizedBlock(SGQuantMixin):
            # NOTE: if **kwargs are existing, then quant descriptors are passed there!
            # NOTE: because we don't override `from_float`,
            #       then the float instance should have `in_feats` and `out_feats` as state
            def __init__(self, in_feats, out_feats, **kwargs) -> None:
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear = quant_nn.QuantLinear(
                    in_feats,
                    out_feats,
                    quant_desc_input=kwargs["quant_desc_input"],
                    quant_desc_weight=kwargs["quant_desc_weight"],
                )

            def forward(self, x):
                return self.linear(self.flatten(x))

        class MyModel(nn.Module):
            def __init__(self, res, n_classes) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
                self.my_block = QuantizedMapping(
                    float_module=MyBlock(4 * (res**2), n_classes),
                    quantized_target_class=MyQuantizedBlock,
                    input_quant_descriptor=QuantDescriptor(calib_method="max"),
                    weights_quant_descriptor=QuantDescriptor(calib_method="histogram"),
                )

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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))
        self.assertTrue(type(module.my_block.linear._input_quantizer._calibrator) == MaxCalibrator)
        self.assertTrue(type(module.my_block.linear._weight_quantizer._calibrator) == HistogramCalibrator)

    def test_quant_descriptors_are_piped_to_custom_quant_modules_if_expects_in_init(self):
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

        class MyQuantizedBlock(SGQuantMixin):
            # NOTE: `since quant_desc_input`, `quant_desc_weight` are existing, then quant descriptors are passed there!
            # NOTE: because we don't override `from_float`,
            #       then the float instance should have `in_feats` and `out_feats` as state
            def __init__(self, in_feats, out_feats, quant_desc_input, quant_desc_weight) -> None:
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear = quant_nn.QuantLinear(
                    in_feats,
                    out_feats,
                    quant_desc_input=quant_desc_input,
                    quant_desc_weight=quant_desc_weight,
                )

            def forward(self, x):
                return self.linear(self.flatten(x))

        class MyModel(nn.Module):
            def __init__(self, res, n_classes) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
                self.my_block = QuantizedMapping(
                    float_module=MyBlock(4 * (res**2), n_classes),
                    quantized_target_class=MyQuantizedBlock,
                    input_quant_descriptor=QuantDescriptor(calib_method="max"),
                    weights_quant_descriptor=QuantDescriptor(calib_method="histogram"),
                )

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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))
        self.assertTrue(type(module.my_block.linear._input_quantizer._calibrator) == MaxCalibrator)
        self.assertTrue(type(module.my_block.linear._weight_quantizer._calibrator) == HistogramCalibrator)

    def test_quant_descriptors_are_not_piped_if_custom_quant_module_does_not_expect_them(self):
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

        class MyQuantizedBlock(SGQuantMixin):
            # NOTE: because we don't override `from_float`,
            #       then the float instance should have `in_feats` and `out_feats` as state
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
                self.my_block = QuantizedMapping(float_module=MyBlock(4 * (res**2), n_classes), quantized_target_class=MyQuantizedBlock)

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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))

    def test_custom_quantized_mappings_are_recursively_quantized_if_required(self):
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

        class MyQuantizedBlock(SGQuantMixin):
            def __init__(self, in_feats, out_feats) -> None:
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(in_feats, out_feats)

            def forward(self, x):
                return self.linear(self.flatten(x))

        class MyModel(nn.Module):
            def __init__(self, res, n_classes) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
                self.my_block = QuantizedMapping(
                    float_module=MyBlock(4 * (res**2), n_classes),
                    quantized_target_class=MyQuantizedBlock,
                    action=QuantizedMetadata.ReplacementAction.REPLACE_AND_RECURE,
                )

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

        # ASSERT
        with torch.no_grad():
            y = module(x)
            torch.testing.assert_close(y.size(), (1, n_clss))

        self.assertTrue(isinstance(module.conv, SelectiveQuantizer.mapping_instructions[nn.Conv2d].quantized_target_class))
        self.assertTrue(isinstance(module.my_block, MyQuantizedBlock))
        self.assertTrue(isinstance(module.my_block.linear, SelectiveQuantizer.mapping_instructions[nn.Linear].quantized_target_class))

    def test_torchvision_resnet_sg_vanilla_quantization_matches_pytorch_quantization(self):
        resnet_sg = torchvision.models.resnet50(pretrained=True)

        # SG SELECTIVE QUANTIZATION
        sq = SelectiveQuantizer(
            custom_mappings={
                torch.nn.Conv2d: QuantizedMetadata(
                    torch.nn.Conv2d,
                    quant_nn.QuantConv2d,
                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                    input_quant_descriptor=QuantDescriptor(calib_method="histogram"),
                    weights_quant_descriptor=QuantDescriptor(calib_method="max", axis=0),
                ),
                torch.nn.Linear: QuantizedMetadata(
                    torch.nn.Linear,
                    quant_nn.QuantLinear,
                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                    input_quant_descriptor=QuantDescriptor(calib_method="histogram"),
                    weights_quant_descriptor=QuantDescriptor(calib_method="max", axis=0),
                ),
                torch.nn.AdaptiveAvgPool2d: QuantizedMetadata(
                    torch.nn.AdaptiveAvgPool2d,
                    quant_nn.QuantAdaptiveAvgPool2d,
                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                    input_quant_descriptor=QuantDescriptor(calib_method="max"),
                ),
            },
            default_per_channel_quant_weights=True,
        )

        sq.quantize_module(resnet_sg, preserve_state_dict=True)

        # PYTORCH-QUANTIZATION
        quant_desc_input = QuantDescriptor(calib_method="histogram")
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

        quant_modules.initialize()
        resnet_pyquant = torchvision.models.resnet50(pretrained=True)
        quant_modules.deactivate()

        for (n1, p1), (n2, p2) in zip(resnet_sg.named_parameters(), resnet_pyquant.named_parameters()):
            torch.testing.assert_allclose(p1, p2)

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            y_pyquant = resnet_pyquant(x)
            y_sg = resnet_sg(x)
            torch.testing.assert_close(y_sg, y_pyquant)

    def test_sg_resnet_sg_vanilla_quantization_matches_pytorch_quantization(self):
        # SG SELECTIVE QUANTIZATION
        from super_gradients.training.models.classification_models.resnet import Bottleneck

        sq = SelectiveQuantizer(
            custom_mappings={
                torch.nn.Conv2d: QuantizedMetadata(
                    torch.nn.Conv2d,
                    quant_nn.QuantConv2d,
                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                    input_quant_descriptor=QuantDescriptor(calib_method="histogram"),
                    weights_quant_descriptor=QuantDescriptor(calib_method="max", axis=0),
                ),
                torch.nn.Linear: QuantizedMetadata(
                    torch.nn.Linear,
                    quant_nn.QuantLinear,
                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                    input_quant_descriptor=QuantDescriptor(calib_method="histogram"),
                    weights_quant_descriptor=QuantDescriptor(calib_method="max", axis=0),
                ),
                torch.nn.AdaptiveAvgPool2d: QuantizedMetadata(
                    torch.nn.AdaptiveAvgPool2d,
                    quant_nn.QuantAdaptiveAvgPool2d,
                    action=QuantizedMetadata.ReplacementAction.REPLACE,
                    input_quant_descriptor=QuantDescriptor(calib_method="max"),
                ),
            },
            default_per_channel_quant_weights=True,
        )

        # SG registers non-naive QuantBottleneck that will have different behaviour, pop it for testing purposes
        if Bottleneck in sq.mapping_instructions:
            sq.mapping_instructions.pop(Bottleneck)

        resnet_sg: nn.Module = super_gradients.training.models.get(Models.RESNET50, pretrained_weights="imagenet", num_classes=1000)
        sq.quantize_module(resnet_sg, preserve_state_dict=True)

        # PYTORCH-QUANTIZATION
        quant_desc_input = QuantDescriptor(calib_method="histogram")
        quant_desc_weights = QuantDescriptor(calib_method="max", axis=0)

        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weights)

        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weights)

        quant_nn.QuantAdaptiveAvgPool2d.set_default_quant_desc_input(QuantDescriptor(calib_method="histogram"))

        quant_modules.initialize()
        resnet_pyquant: nn.Module = super_gradients.training.models.get(Models.RESNET50, pretrained_weights="imagenet", num_classes=1000)

        quant_modules.deactivate()

        for (n1, p1), (n2, p2) in zip(resnet_sg.named_parameters(), resnet_pyquant.named_parameters()):
            torch.testing.assert_allclose(p1, p2)

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            y_pyquant = resnet_pyquant(x)
            y_sg = resnet_sg(x)
            torch.testing.assert_close(y_sg, y_pyquant)


if __name__ == "__main__":
    unittest.main()
