import os
import unittest

from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, utils as core_utils, models
from super_gradients.training.dataloaders.dataloaders import coco2017_val
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionVisualization


class TestDetectionUtils(unittest.TestCase):
    def test_visualization(self):

        # Create Yolo model
        trainer = Trainer("visualization_test")
        model = models.get(Models.YOLOX_N, pretrained_weights="coco")
        post_prediction_callback = YoloPostPredictionCallback()

        # Simulate one iteration of validation subset
        valid_loader = coco2017_val()
        batch_i, (imgs, targets) = 0, next(iter(valid_loader))
        imgs = core_utils.tensor_container_to_device(imgs, trainer.device)
        targets = core_utils.tensor_container_to_device(targets, trainer.device)
        output = model(imgs)
        output = post_prediction_callback(output)
        # Visualize the batch
        DetectionVisualization.visualize_batch(imgs, output, targets, batch_i, COCO_DETECTION_CLASSES_LIST, trainer.checkpoints_dir_path)

        # Assert images ware created and delete them
        img_name = "{}/{}_{}.jpg"
        for i in range(4):
            img_path = img_name.format(trainer.checkpoints_dir_path, batch_i, i)
            self.assertTrue(os.path.exists(img_path))
            os.remove(img_path)


if __name__ == "__main__":
    unittest.main()
