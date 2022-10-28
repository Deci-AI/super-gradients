import unittest

import torch

from super_gradients.training.models.detection_models.csp_resnet import CSPResNet


class TestPPYOLOE(unittest.TestCase):
    def test_cspresnet_creation(self):
        ppyoloe = CSPResNet()
        dummy_input = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            feature_maps = ppyoloe(dummy_input)
            self.assertEqual(len(feature_maps), 3)

        ppyoloe_plus = CSPResNet(use_alpha=True)
        dummy_input = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            feature_maps = ppyoloe_plus(dummy_input)
            self.assertEqual(len(feature_maps), 3)


if __name__ == "__main__":
    unittest.main()
