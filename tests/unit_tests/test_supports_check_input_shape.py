import unittest

import torch

from super_gradients.common.object_names import Models
from super_gradients.training import models


class TestSupportsInputShapeCheck(unittest.TestCase):
    def setUp(self):
        self.models_to_check = [Models.YOLO_NAS_S, Models.YOLO_NAS_POSE_S, Models.PP_LITE_T_SEG50, Models.STDC1_SEG50, Models.DDRNET_23]

    @torch.no_grad()
    def test_can_run_inference_with_min_size(self):
        for model in self.models_to_check:
            with self.subTest(model=model):
                model = models.get(model, num_classes=20).eval()
                min_shape = model.get_minimum_input_shape_size()
                if min_shape is not None:
                    dummy_input = torch.randn(1, 3, *min_shape)
                    model.validate_input_shape(dummy_input.size())
                    model(dummy_input)

    @torch.no_grad()
    def test_validate_invalid_size(self):
        for model in self.models_to_check:
            with self.subTest(model=model):
                model = models.get(model, num_classes=20).eval()
                steps = model.get_input_shape_steps()
                invalid_shape = [x * 4 + 1 for x in steps]
                dummy_input = torch.randn(1, 3, *invalid_shape)
                with self.assertRaises(ValueError):
                    model.validate_input_shape(dummy_input.size())


if __name__ == "__main__":
    unittest.main()
