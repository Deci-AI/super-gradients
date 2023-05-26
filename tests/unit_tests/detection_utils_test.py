import os
import unittest

import numpy as np
import torch.cuda

from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, utils as core_utils, models
from super_gradients.training.dataloaders.dataloaders import coco2017_val
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.metrics import DetectionMetrics, DetectionMetrics_050
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionVisualization
from tests.core_test_utils import is_data_available


class TestDetectionUtils(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.get(Models.YOLOX_N, pretrained_weights="coco").to(self.device)
        self.model.eval()

    @unittest.skipIf(not is_data_available(), "run only when /data is available")
    def test_visualization(self):

        valid_loader = coco2017_val(dataloader_params={"batch_size": 16, "num_workers": 0})
        trainer = Trainer("visualization_test")
        post_prediction_callback = YoloPostPredictionCallback()

        # Simulate one iteration of validation subset
        batch_i, batch = 0, next(iter(valid_loader))
        imgs, targets = batch[:2]
        imgs = core_utils.tensor_container_to_device(imgs, self.device)
        targets = core_utils.tensor_container_to_device(targets, self.device)
        output = self.model(imgs)
        output = post_prediction_callback(output)
        # Visualize the batch
        DetectionVisualization.visualize_batch(imgs, output, targets, batch_i, COCO_DETECTION_CLASSES_LIST, trainer.checkpoints_dir_path)

        # Assert images ware created and delete them
        img_name = "{}/{}_{}.jpg"
        for i in range(4):
            img_path = img_name.format(trainer.checkpoints_dir_path, batch_i, i)
            self.assertTrue(os.path.exists(img_path))
            os.remove(img_path)

    @unittest.skipIf(not is_data_available(), "run only when /data is available")
    def test_detection_metrics(self):

        valid_loader = coco2017_val(dataloader_params={"batch_size": 16, "num_workers": 0})

        metrics = [
            DetectionMetrics(num_cls=80, post_prediction_callback=YoloPostPredictionCallback(), normalize_targets=True),
            DetectionMetrics_050(num_cls=80, post_prediction_callback=YoloPostPredictionCallback(), normalize_targets=True),
            DetectionMetrics(num_cls=80, post_prediction_callback=YoloPostPredictionCallback(conf=2), normalize_targets=True),
        ]

        ref_values = [
            np.array([0.24701539, 0.40294355, 0.34654024, 0.28485271]),
            np.array([0.34666198, 0.56854934, 0.5079478, 0.40414381]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ]

        for met, ref_val in zip(metrics, ref_values):
            met.reset()
            for i, (imgs, targets, extras) in enumerate(valid_loader):
                if i > 5:
                    break
                imgs = core_utils.tensor_container_to_device(imgs, self.device)
                targets = core_utils.tensor_container_to_device(targets, self.device)
                output = self.model(imgs)
                met.update(output, targets, device=self.device, inputs=imgs)
            results = met.compute()
            values = np.array([x.item() for x in list(results.values())])
            self.assertTrue(np.allclose(values, ref_val, rtol=1e-3, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
