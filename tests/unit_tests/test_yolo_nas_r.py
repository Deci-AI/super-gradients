import unittest

import torch
from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_post_prediction_callback import rboxes_nms


class TestYoloNasR(unittest.TestCase):
    def test_rboxes_nms(self):
        boxes = torch.rand([2, 5])
        boxes[:, 2:] = torch.abs(boxes[:, 2:])
        scores = torch.rand([2])
        keep = rboxes_nms(boxes, scores, 0.5)
        print(keep)


if __name__ == "__main__":
    unittest.main()
