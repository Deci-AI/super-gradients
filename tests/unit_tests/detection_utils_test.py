import os
import tempfile
import unittest

import numpy as np
import torch.cuda

from super_gradients.common.object_names import Models
from super_gradients.training import utils as core_utils, models
from super_gradients.training.dataloaders.dataloaders import coco2017_val
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import flat_collate_tensors_with_batch_index
from super_gradients.training.metrics import DetectionMetrics, DetectionMetrics_050
from super_gradients.training.models.detection_models.yolo_base import YoloXPostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionVisualization, DetectionPostPredictionCallback, xyxy2cxcywh
from tests.core_test_utils import is_data_available


class TestDetectionUtils(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.get(Models.YOLOX_N, pretrained_weights="coco").to(self.device)
        self.model.eval()

    def test_detection_metric_with_calc_best_score_thresholds(self):
        class DummyCallback(DetectionPostPredictionCallback):
            def forward(self, p, device=None):
                return p

        class_names = ["A", "B", "C"]
        num_classes = len(class_names)
        metric = DetectionMetrics(
            num_cls=num_classes,
            post_prediction_callback=DummyCallback(),
            normalize_targets=True,
            calc_best_score_thresholds=True,
            include_classwise_ap=True,
            class_names=class_names,
        )

        # x1, y1, x2, y2, confidence, class_label

        num_predictions = 100
        num_targets = 64
        preds = torch.cat(
            [
                torch.randint(0, 100, (num_predictions, 2)),  # [x1,y1]
                torch.randint(100, 200, (num_predictions, 2)),  # [x2,y2]
                torch.randn((num_predictions, 1)).sigmoid(),
                torch.randint(0, num_classes, (num_predictions, 1)),  # [x2,y2]
            ],
            dim=-1,
        ).float()

        targets = torch.cat(
            [
                torch.randint(0, num_classes, (num_targets, 1)),  # [x2,y2]
                torch.randint(0, 100, (num_targets, 2)),  # [x1,y1]
                torch.randint(100, 200, (num_targets, 2)),  # [x2,y2]
            ],
            dim=-1,
        ).float()

        targets[:, 1:] = xyxy2cxcywh(targets[:, 1:])
        targets_flat = flat_collate_tensors_with_batch_index([targets])

        metric(preds=[preds], target=targets_flat, device="cpu", inputs=torch.zeros((1, 3, 640, 640)))
        metric_values = metric.compute()
        self.assertTrue("Best_score_threshold" in metric_values)
        for metric_value_name in metric.best_threshold_per_class_names:
            self.assertTrue(metric_value_name in metric_values)

    @unittest.skipIf(not is_data_available(), "run only when /data is available")
    def test_visualization(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            valid_loader = coco2017_val(dataloader_params={"batch_size": 16, "num_workers": 0})
            post_prediction_callback = YoloXPostPredictionCallback()

            # Simulate one iteration of validation subset
            batch_i, batch = 0, next(iter(valid_loader))
            imgs, targets = batch[:2]
            imgs = core_utils.tensor_container_to_device(imgs, self.device)
            targets = core_utils.tensor_container_to_device(targets, self.device)
            output = self.model(imgs)
            output = post_prediction_callback(output)
            # Visualize the batch
            DetectionVisualization.visualize_batch(imgs, output, targets, batch_i, COCO_DETECTION_CLASSES_LIST, tmpdirname)

            # Assert images ware created and delete them
            img_name = "{}/{}_{}.jpg"
            for i in range(4):
                img_path = img_name.format(tmpdirname, batch_i, i)
                self.assertTrue(os.path.exists(img_path))
                os.remove(img_path)

    @unittest.skipIf(not is_data_available(), "run only when /data is available")
    def test_detection_metrics(self):
        valid_loader = coco2017_val(dataloader_params={"batch_size": 16, "num_workers": 0})

        metrics = [
            DetectionMetrics(num_cls=80, post_prediction_callback=YoloXPostPredictionCallback(), normalize_targets=True),
            DetectionMetrics_050(num_cls=80, post_prediction_callback=YoloXPostPredictionCallback(), normalize_targets=True),
            DetectionMetrics(num_cls=80, post_prediction_callback=YoloXPostPredictionCallback(conf=2), normalize_targets=True),
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
            for expected, actual in zip(ref_val, values):
                self.assertAlmostEqual(expected, actual, delta=5e-3)


if __name__ == "__main__":
    unittest.main()
