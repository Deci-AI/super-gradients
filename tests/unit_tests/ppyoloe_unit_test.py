import unittest

import torch

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
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

    def test_ppyoloe_batched_vs_sequential_loss(self):
        for use_static_assigner in [True, False]:
            with self.subTest(use_static_assigner=use_static_assigner):
                torch.random.manual_seed(0)
                batched_loss = PPYoloELoss(
                    num_classes=80, use_varifocal_loss=True, use_static_assigner=use_static_assigner, reg_max=16, use_batched_assignment=True
                )
                sequential_loss = PPYoloELoss(
                    num_classes=80, use_varifocal_loss=True, use_static_assigner=use_static_assigner, reg_max=16, use_batched_assignment=False
                )

                model = models.get(Models.PP_YOLOE_S, num_classes=80)
                random_input = torch.randn(4, 3, 640, 480)
                output = model(random_input)

                # (N, 6) (batch_index, class_index, cx, cy, w, h)
                # Five objects in the first image, three objects in the second image, two objects in the third image, no objects in the fourth image
                targets = torch.tensor(
                    [
                        [0, 2, 40, 60, 100, 200],
                        [0, 3, 100, 200, 100, 200],
                        [0, 4, 200, 300, 100, 200],
                        [0, 5, 300, 400, 100, 200],
                        [0, 6, 400, 500, 100, 200],
                        [1, 2, 40, 60, 100, 200],
                        [1, 3, 100, 200, 100, 200],
                        [1, 4, 200, 300, 100, 200],
                        [2, 2, 40, 60, 100, 200],
                        [2, 3, 100, 200, 100, 200],
                    ]
                ).float()

                batched_loss_output = batched_loss(output, targets)
                sequential_loss_output = sequential_loss(output, targets)

                self.assertAlmostEqual(batched_loss_output[0].item(), sequential_loss_output[0].item(), places=4)
                self.assertAlmostEqual(batched_loss_output[1][0].item(), sequential_loss_output[1][0].item(), places=4)
                self.assertAlmostEqual(batched_loss_output[1][1].item(), sequential_loss_output[1][1].item(), places=4)
                self.assertAlmostEqual(batched_loss_output[1][2].item(), sequential_loss_output[1][2].item(), places=4)
                self.assertAlmostEqual(batched_loss_output[1][3].item(), sequential_loss_output[1][3].item(), places=4)


if __name__ == "__main__":
    unittest.main()
