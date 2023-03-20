import unittest
from super_gradients.common.registry.registry import ARCHITECTURES
from super_gradients.training.models.classification_models.repvgg import RepVggA1
from super_gradients.training.utils.utils import HpmStruct
import torch
import copy


class BackboneBasedModel(torch.nn.Module):
    """
    Auxiliary model which will use repvgg as backbone
    """

    def __init__(self, backbone, backbone_output_channel, num_classes=1000):
        super(BackboneBasedModel, self).__init__()
        self.backbone = backbone
        self.conv = torch.nn.Conv2d(in_channels=backbone_output_channel, out_channels=backbone_output_channel, kernel_size=1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(backbone_output_channel)  # Adding a bn layer that should NOT be fused
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = torch.nn.Linear(backbone_output_channel, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def prep_model_for_conversion(self):
        if hasattr(self.backbone, "prep_model_for_conversion"):
            self.backbone.prep_model_for_conversion()


class TestRepVgg(unittest.TestCase):
    def setUp(self):
        # contains all arch_params needed for initialization of all architectures
        self.all_arch_params = HpmStruct(**{"num_classes": 10, "width_mult": 1, "build_residual_branches": True})

        self.backbone_arch_params = copy.deepcopy(self.all_arch_params)
        self.backbone_arch_params.override(backbone_mode=True)

    def test_deployment_architecture(self):
        """
        Validate all models that has a deployment mode are in fact different after deployment
        """
        image_size = 224
        in_channels = 3
        for arch_name in ARCHITECTURES:
            # skip custom constructors to keep all_arch_params as general as a possible
            if "repvgg" not in arch_name or "custom" in arch_name:
                continue
            model = ARCHITECTURES[arch_name](arch_params=self.all_arch_params)
            self.assertTrue(hasattr(model.stem, "branch_3x3"))  # check single layer for training mode
            self.assertTrue(model.build_residual_branches)

            training_mode_sd = model.state_dict()
            for module in training_mode_sd:
                self.assertFalse("reparam" in module)  # deployment block included in training mode
            test_input = torch.ones((1, in_channels, image_size, image_size))
            model.eval()
            training_mode_output = model(test_input)

            model.prep_model_for_conversion()
            self.assertTrue(hasattr(model.stem, "rbr_reparam"))  # check single layer for training mode
            self.assertFalse(model.build_residual_branches)

            deployment_mode_sd = model.state_dict()
            for module in deployment_mode_sd:
                self.assertFalse("running_mean" in module)  # BN were not fused
                self.assertFalse("branch" in module)  # branches were not joined

            deployment_mode_output = model(test_input)
            # difference is of very low magnitude
            self.assertFalse(False in torch.isclose(training_mode_output, deployment_mode_output, atol=1e-4))

    def test_backbone_mode(self):
        """
        Validate repvgg models (A1) as backbone.
        """
        image_size = 224
        in_channels = 3
        test_input = torch.rand((1, in_channels, image_size, image_size))
        backbone_model = RepVggA1(self.backbone_arch_params)
        model = BackboneBasedModel(backbone_model, backbone_output_channel=1280, num_classes=self.backbone_arch_params.num_classes)

        backbone_model.eval()
        model.eval()

        backbone_training_mode_output = backbone_model(test_input)
        model_training_mode_output = model(test_input)
        # check that the linear head was dropped
        self.assertFalse(backbone_training_mode_output.shape[1] == self.backbone_arch_params.num_classes)

        training_mode_sd = model.state_dict()
        for module in training_mode_sd:
            self.assertFalse("reparam" in module)  # deployment block included in training mode

        model.prep_model_for_conversion()
        deployment_mode_sd_list = list(model.state_dict().keys())
        self.assertTrue("bn.running_mean" in deployment_mode_sd_list)  # Verify non backbone batch norm wasn't fused
        for module in deployment_mode_sd_list:
            self.assertFalse("running_mean" in module and module.startswith("backbone"))  # BN were not fused
            self.assertFalse("branch" in module and module.startswith("backbone"))  # branches were not joined
        model_deployment_mode_output = model(test_input)
        self.assertFalse(False in torch.isclose(model_deployment_mode_output, model_training_mode_output, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
