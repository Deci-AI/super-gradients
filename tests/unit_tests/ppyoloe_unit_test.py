import os.path
import unittest

import hydra
import pkg_resources
import torch
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from super_gradients.training.models.detection_models.csp_resnet import CSPResNet
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloE
from super_gradients.training.utils.hydra_utils import normalize_path


class TestPPYOLOE(unittest.TestCase):
    def get_model_arch_params(self, config_name):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir)):
            cfg = compose(config_name=normalize_path(config_name))
            cfg = hydra.utils.instantiate(cfg)
            arch_params = cfg.arch_params

        return arch_params

    def _test_csp_resnet_variant(self, variant):
        arch_params = self.get_model_arch_params(os.path.join("arch_params", variant))
        ppyoloe = CSPResNet(**arch_params)
        dummy_input = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            feature_maps = ppyoloe(dummy_input)
            self.assertEqual(len(feature_maps), 3)

    def _test_ppyoloe_variant(self, variant):
        arch_params = self.get_model_arch_params(os.path.join("arch_params", variant))
        ppyoloe = PPYoloE(**arch_params).eval()
        dummy_input = torch.randn(1, 3, 640, 480)
        with torch.no_grad():
            feature_maps = ppyoloe(dummy_input)
            self.assertIsNotNone(feature_maps)

    def test_csp_resnet_s(self):
        self._test_csp_resnet_variant("csp_resnet_l_arch_params")

    def test_csp_resnet_m(self):
        self._test_csp_resnet_variant("csp_resnet_m_arch_params")

    def test_csp_resnet_l(self):
        self._test_csp_resnet_variant("csp_resnet_l_arch_params")

    def test_csp_resnet_x(self):
        self._test_csp_resnet_variant("csp_resnet_x_arch_params")

    def test_ppyoloe_s(self):
        self._test_ppyoloe_variant("ppyoloe_s_arch_params")

    def test_ppyoloe_m(self):
        self._test_ppyoloe_variant("ppyoloe_m_arch_params")

    def test_ppyoloe_l(self):
        self._test_ppyoloe_variant("ppyoloe_l_arch_params")

    def test_ppyoloe_x(self):
        self._test_ppyoloe_variant("ppyoloe_x_arch_params")


if __name__ == "__main__":
    unittest.main()
