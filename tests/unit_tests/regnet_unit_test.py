import torch
import torch.nn as nn
import unittest

from super_gradients.training.models.classification_models.regnet import (
    CustomRegNet,
    NASRegNet,
    RegNetY200,
    RegNetY400,
    RegNetY600,
    RegNetY800,
    Stem,
    Stage,
    XBlock,
)
from super_gradients.training.utils.utils import HpmStruct


class TestRegnet(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.arch_params = HpmStruct(**{"num_classes": 1000})

    @staticmethod
    def verify_2_archs_are_identical(model_1: nn.Module, model_2: nn.Module):
        state_dict_1 = model_1.state_dict()
        model_2.load_state_dict(state_dict_1, strict=False)

    def test_custom_and_nas_regnet_can_build_regnetY200(self):
        """Test that when build Nas Regnet and Custom Regnet with the correct params - they build RegnetY200"""
        regnet_y_200 = RegNetY200(arch_params=self.arch_params)
        # Parameters identical to regnet_y_200
        nas_regnet = NASRegNet(arch_params=HpmStruct(**{"structure": [24, 36, 2.5, 13, 1, 8, 2, 4], "num_classes": 1000}))
        regnet_y_200_arch_params = {
            "initial_width": 24,
            "slope": 36,
            "quantized_param": 2.5,
            "network_depth": 13,
            "bottleneck_ratio": 1,
            "group_width": 8,
            "stride": 2,
            "num_classes": 1000,
        }
        custom_regnet = CustomRegNet(arch_params=HpmStruct(**regnet_y_200_arch_params))

        self.verify_2_archs_are_identical(regnet_y_200, nas_regnet)
        self.verify_2_archs_are_identical(regnet_y_200, custom_regnet)

    def test_regnet_model_creation(self):
        """
        Tests that the basic Regnets can be created
        """
        dummy_input = torch.randn(1, 3, 224, 224)

        regnet_y_200 = RegNetY200(arch_params=self.arch_params)
        regnet_y_400 = RegNetY400(arch_params=self.arch_params)
        regnet_y_600 = RegNetY600(arch_params=self.arch_params)
        regnet_y_800 = RegNetY800(arch_params=self.arch_params)

        with torch.no_grad():
            for model in [regnet_y_200, regnet_y_400, regnet_y_600, regnet_y_800]:
                output = model(dummy_input)
                self.assertIsNotNone(output)

    def test_dropout_forward_backward(self):
        """
        Test that output is stochastic in training and is fixed in eval with Dropout.
        """
        arch_params = HpmStruct(**{"num_classes": 1000, "dropout_prob": 0.3})
        model = RegNetY200(arch_params=arch_params)
        dummy_input = torch.randn(1, 3, 224, 224)

        model.train()
        self.assertFalse(torch.equal(model(dummy_input), model(dummy_input)))

        model.eval()
        self.assertTrue(torch.equal(model(dummy_input), model(dummy_input)))

    def test_droppath_forward_backward(self):
        """
        Test that output is stochastic in training and is fixed in eval with DropPath.
        """
        arch_params = HpmStruct(**{"num_classes": 1000, "droppath_prob": 0.2})
        model = RegNetY200(arch_params=arch_params)
        dummy_input = torch.randn(1, 3, 224, 224)

        model.train()
        self.assertFalse(torch.equal(model(dummy_input), model(dummy_input)))

        model.eval()
        self.assertTrue(torch.equal(model(dummy_input), model(dummy_input)))

    def test_nas_regnet_logic_is_backward_competible(self):
        """
        Runs several configurations of CustomRegnet models and validates that the logic wasn't change in the Regnet class
        This is important in order to reproduce previous Deci models and be backward competible
        """
        # THE LIST CONSISTS SEVERAL CUSTOM REGNET "ENCODINGS" AND THE CORRESPONDING XBLOCK STRUCTURE OF THE MODEL
        selected_arch_and_corresponding_configs = [
            {"struct": [56, 10, 2.2, 8, 2, 8, 2, 0], "expected_config": [3, 32, 32, 32, 16, 16, 16, 2, (2, 2), None, 32, 32, 32, 32, 4, (2, 2), None]},
            {"struct": [56, 10, 2.3, 11, 1, 1, 3, 0], "expected_config": [3, 32, 32, 32, 56, 56, 56, 56, (3, 3), None, 56, 128, 128, 128, 128, (3, 3), None]},
            {
                "struct": [70, 20, 2.6, 13, 0.5, 16, 2, 4],
                "expected_config": [
                    3,
                    32,
                    32,
                    32,
                    288,
                    288,
                    288,
                    18,
                    (2, 2),
                    nn.Module,
                    144,
                    736,
                    736,
                    736,
                    46,
                    (2, 2),
                    nn.Module,
                    368,
                    1888,
                    1888,
                    1888,
                    118,
                    (2, 2),
                    nn.Module,
                ],
            },
            {
                "struct": [8, 20, 2.3, 13, 0.16666666666666666, 1, 2, 2],
                "expected_config": [
                    3,
                    32,
                    32,
                    32,
                    288,
                    288,
                    288,
                    288,
                    (2, 2),
                    nn.Module,
                    48,
                    1440,
                    1440,
                    1440,
                    1440,
                    (2, 2),
                    nn.Module,
                    240,
                    3456,
                    3456,
                    3456,
                    3456,
                    (2, 2),
                    nn.Module,
                    576,
                    8064,
                    8064,
                    8064,
                    8064,
                    (2, 2),
                    nn.Module,
                ],
            },
            {"struct": [56, 10, 2.4, 13, 2, 8, 1, 0], "expected_config": [3, 32, 32, 32, 16, 16, 16, 2, (1, 1), None, 32, 32, 32, 32, 4, (1, 1), None]},
        ]

        for arch_conf_pair in selected_arch_and_corresponding_configs:
            expected_config = iter(arch_conf_pair["expected_config"])
            model = NASRegNet(HpmStruct(**{"structure": arch_conf_pair["struct"], "num_classes": 1000}))

            for stage in model.net.children():
                # CHECK CORRECTNESS OF THE STEM
                if isinstance(stage, Stem):
                    assert stage.conv.in_channels == next(expected_config)
                    assert stage.conv.out_channels == next(expected_config)
                    assert stage.bn.num_features == next(expected_config)
                # CHECK THE CORRECTNESS OF THE FIRST XBlock IN EACH STAGE
                if isinstance(stage, Stage):
                    for block in stage.blocks.children():
                        if isinstance(block, XBlock):
                            assert block.conv_block_1[0].in_channels == next(expected_config)
                            assert block.conv_block_1[0].out_channels == next(expected_config)
                            assert block.conv_block_2[0].in_channels == next(expected_config)
                            assert block.conv_block_2[0].out_channels == next(expected_config)
                            assert block.conv_block_2[0].groups == next(expected_config)
                            assert block.conv_block_2[0].stride == next(expected_config)
                            se_block = next(expected_config)
                            assert block.se is None if se_block is None else isinstance(block, se_block)
                            # SKIP TO THE NEXT STAGE
                            break


if __name__ == "__main__":
    unittest.main()
