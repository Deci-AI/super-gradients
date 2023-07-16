import unittest

import torch

from super_gradients.training import models
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback


class TestPostPredictionCallback(unittest.TestCase):
    def _default_yolo_post_prediction_callback(self):
        """
        Use low confidence to force a non-empty nms result.
        """
        return YoloPostPredictionCallback(conf=1e-6)

    def _default_mock_decoded_output(self):
        """
        mock output tensor after a final decode module, i.e DetectX, with shapes [B, Num anchors, 5 + num_classes]
        """
        return torch.cat([torch.randn(1, 500, 4), torch.sigmoid(torch.randn(1, 500, 81))], dim=2)  # localization  # classification scores

    def test_yolo_post_prediction_callback_single_input(self):
        callback = self._default_yolo_post_prediction_callback()

        mock_single_model_output = self._default_mock_decoded_output()
        _ = callback(mock_single_model_output)

    def test_yolo_post_prediction_callback_multiple_input(self):
        callback = self._default_yolo_post_prediction_callback()

        mock_multiple_model_outputs = [self._default_mock_decoded_output(), [torch.randn(1, 1, 10, 10, 85), torch.randn(1, 1, 20, 20, 85)]]  # mock logits
        # sanity check multiple input as list
        _ = callback(mock_multiple_model_outputs)
        # sanity check multiple input as tuple
        _ = callback(tuple(mock_multiple_model_outputs))

    def test_yolo_post_prediction_callback_yolox_output(self):
        """
        Sanity check for yolox usage with YoloPostPredictionCallback.
        """
        callback = self._default_yolo_post_prediction_callback()
        model = models.get(model_name="yolox_s", num_classes=80).eval()

        x = torch.randn(1, 3, 320, 320)
        output = model(x)
        _ = callback(output)


if __name__ == "__main__":
    unittest.main()
