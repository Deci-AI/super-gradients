import unittest

import torch

import super_gradients
from super_gradients.common.object_names import Models
from super_gradients.training import models


class ReplaceHeadUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        super_gradients.init_trainer()

    def test_yolox_replace_head(self):
        input = torch.randn(1, 3, 640, 640).to(self.device)
        for model in [Models.YOLOX_S, Models.YOLOX_M, Models.YOLOX_L, Models.YOLOX_T]:
            model = models.get(model, pretrained_weights="coco").to(self.device).eval()
            num_classes = 100
            model.replace_head(new_num_classes=num_classes)
            outputs = model.forward(input)
            self.assertEqual(outputs[0].size(4), num_classes + 5)
            self.assertEqual(outputs[1].size(4), num_classes + 5)
            self.assertEqual(outputs[2].size(4), num_classes + 5)

    def test_ppyolo_replace_head(self):
        input = torch.randn(1, 3, 640, 640).to(self.device)
        for model in [Models.PP_YOLOE_S, Models.PP_YOLOE_M, Models.PP_YOLOE_L, Models.PP_YOLOE_X]:
            model = models.get(model, pretrained_weights="coco").to(self.device).eval()
            model.replace_head(new_num_classes=100)
            (_, pred_scores), _ = model.forward(input)
            self.assertEqual(pred_scores.size(2), 100)

    def test_yolo_nas_replace_head(self):
        input = torch.randn(1, 3, 640, 640).to(self.device)
        for model in [Models.YOLO_NAS_S, Models.YOLO_NAS_M, Models.YOLO_NAS_L]:
            model = models.get(model, pretrained_weights="coco").to(self.device).eval()
            model.replace_head(new_num_classes=100)
            (_, pred_scores), _ = model.forward(input)
            self.assertEqual(pred_scores.size(2), 100)

    def test_dekr_replace_head(self):
        input = torch.randn(1, 3, 640, 640).to(self.device)
        model = models.get(Models.DEKR_W32_NO_DC, num_classes=20, pretrained_weights="coco_pose").to(self.device).eval()
        heatmap, offsets = model.forward(input)
        self.assertEqual(heatmap.size(1), 20 + 1)
        self.assertEqual(offsets.size(1), 20 * 2)


if __name__ == "__main__":
    unittest.main()
