import unittest
from pathlib import Path

from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val_yolo_nas
from super_gradients.training import Trainer
from super_gradients.training.models.detection_models.sliding_window_detection_forward_wrapper import SlidingWindowInferenceDetectionWrapper
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training import training_hyperparams


class SlidingWindowWrapperTest(unittest.TestCase):
    def setUp(self):
        self.mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "tinycoco")

    def test_train_with_sliding_window_wrapper_validation(self):
        train_params = training_hyperparams.get("coco2017_yolo_nas_s")

        train_params["valid_metrics_list"] = [
            DetectionMetrics(
                normalize_targets=True,
                post_prediction_callback=None,
                num_cls=80,
            )
        ]
        train_params["max_epochs"] = 2
        train_params["lr_warmup_epochs"] = 0
        train_params["lr_cooldown_epochs"] = 0
        train_params["average_best_models"] = False
        train_params["mixed_precision"] = False
        train_params["validation_forward_wrapper"] = SlidingWindowInferenceDetectionWrapper(tile_size=320, tile_step=160, tile_nms_iou=0.65, tile_nms_conf=0.03)

        dl = coco2017_val_yolo_nas(dataset_params=dict(data_dir=self.mini_coco_data_dir))

        trainer = Trainer("test_yolo_nas_s_coco_with_sliding_window")
        model = models.get("yolo_nas_s", num_classes=80, pretrained_weights="coco")
        trainer.train(model=model, training_params=train_params, train_loader=dl, valid_loader=dl)

    def test_yolo_nas_s_coco_with_sliding_window(self):
        trainer = Trainer("test_yolo_nas_s_coco_with_sliding_window")
        model = models.get("yolo_nas_s", num_classes=80, pretrained_weights="coco")
        model = SlidingWindowInferenceDetectionWrapper(tile_size=320, tile_step=160, model=model, tile_nms_iou=0.65, tile_nms_conf=0.03)
        dl = coco2017_val_yolo_nas(dataset_params=dict(data_dir=self.mini_coco_data_dir))
        metric = DetectionMetrics(
            normalize_targets=True,
            post_prediction_callback=None,
            num_cls=80,
        )
        metric_values = trainer.test(model=model, test_loader=dl, test_metrics_list=[metric])
        self.assertAlmostEqual(metric_values[metric.map_str], 0.331, delta=0.001)


if __name__ == "__main__":
    unittest.main()
