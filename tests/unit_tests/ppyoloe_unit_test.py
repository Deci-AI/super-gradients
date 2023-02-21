import unittest

import torch

from super_gradients.training import models
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_e import PPYoloE_X, PPYoloE_S, PPYoloE_M, PPYoloE_L


class TestPPYOLOE(unittest.TestCase):
    def _test_ppyoloe_from_name(self, model_name, pretrained_weights):
        ppyoloe = models.get(model_name, pretrained_weights=pretrained_weights, num_classes=80 if pretrained_weights is None else None).eval()
        dummy_input = torch.randn(1, 3, 640, 480)
        with torch.no_grad():
            feature_maps = ppyoloe(dummy_input)
            self.assertIsNotNone(feature_maps)

    def _test_ppyoloe_from_cls(self, model_cls):
        ppyoloe = model_cls(arch_params={}).eval()
        dummy_input = torch.randn(1, 3, 640, 480)
        with torch.no_grad():
            feature_maps = ppyoloe(dummy_input)
            self.assertIsNotNone(feature_maps)

    def test_ppyoloe_s(self):
        self._test_ppyoloe_from_name("ppyoloe_s", pretrained_weights="coco")
        self._test_ppyoloe_from_cls(PPYoloE_S)

    def test_ppyoloe_m(self):
        self._test_ppyoloe_from_name("ppyoloe_m", pretrained_weights="coco")
        self._test_ppyoloe_from_cls(PPYoloE_M)

    def test_ppyoloe_l(self):
        self._test_ppyoloe_from_name("ppyoloe_l", pretrained_weights=None)
        self._test_ppyoloe_from_cls(PPYoloE_L)

    def test_ppyoloe_x(self):
        self._test_ppyoloe_from_name("ppyoloe_x", pretrained_weights=None)
        self._test_ppyoloe_from_cls(PPYoloE_X)


if __name__ == "__main__":
    unittest.main()
