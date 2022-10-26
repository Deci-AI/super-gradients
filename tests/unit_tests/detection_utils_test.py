import os
import unittest

import torch

from super_gradients.training import Trainer, utils as core_utils, models
from super_gradients.training.dataloaders.dataloaders import coco2017_val
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionVisualization, DetectionFormatConversionAdapter


class TestDetectionUtils(unittest.TestCase):
    def test_visualization(self):

        # Create Yolo model
        trainer = Trainer("visualization_test")
        model = models.get("yolox_n", pretrained_weights="coco")
        post_prediction_callback = YoloPostPredictionCallback()

        # Simulate one iteration of validation subset
        valid_loader = coco2017_val()
        batch_i, (imgs, targets) = 0, next(iter(valid_loader))
        imgs = core_utils.tensor_container_to_device(imgs, trainer.device)
        targets = core_utils.tensor_container_to_device(targets, trainer.device)
        output = model(imgs)
        output = post_prediction_callback(output)
        # Visualize the batch
        DetectionVisualization.visualize_batch(
            imgs, output, targets, batch_i, COCO_DETECTION_CLASSES_LIST, trainer.checkpoints_dir_path
        )

        # Assert images ware created and delete them
        img_name = "{}/{}_{}.jpg"
        for i in range(4):
            img_path = img_name.format(trainer.checkpoints_dir_path, batch_i, i)
            self.assertTrue(os.path.exists(img_path))
            os.remove(img_path)

    def test_output_adapter(self):
        input_boxes = [
            torch.randn((16, 6)),
            torch.randn((0, 6)),  # No bounding boxes
            torch.randn((1, 6)),
            torch.randn((8, 6)),
            torch.randn((13, 6 + 1)),  # Plus one extra channel with additional metadata
        ]

        batch_size = len(input_boxes)
        image_shape = (480, 640)
        output_format = ["XYWH", "CLS_INDEX"]
        adapter = DetectionFormatConversionAdapter(output_format, image_shape)
        outputs = adapter(input_boxes)
        self.assertEqual(len(outputs), batch_size)
        for output in outputs:
            self.assertEqual(len(output), 2)
            self.assertEqual(output[1].size(0), output[0].size(0))
            self.assertEqual(output[0].size(1), 4)
            self.assertEqual(len(output[1].size()), 1)


if __name__ == "__main__":
    unittest.main()
