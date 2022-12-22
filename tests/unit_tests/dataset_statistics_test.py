import unittest

from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders.dataloaders import coco2017_train, coco2017_val
from super_gradients.training.metrics.detection_metrics import DetectionMetrics

from super_gradients.training import Trainer, models
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback


class TestDatasetStatisticsTensorboardLogger(unittest.TestCase):
    def test_dataset_statistics_tensorboard_logger(self):
        """
        ** IMPORTANT NOTE **
        This test is not the usual fail/pass test - it is a visual test. The success criteria is your own visual check
        After launching the test, follow the log the see where was the tensorboard opened. open the tensorboard in your
        browser and make sure the text and plots in the tensorboard are as expected.
        """
        # Create dataset

        trainer = Trainer("dataset_statistics_visual_test")

        model = models.get(Models.YOLOX_S)

        training_params = {
            "max_epochs": 1,  # we dont really need the actual training to run
            "lr_mode": "cosine",
            "initial_lr": 0.01,
            "loss": "yolox_loss",
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 80},
            "dataset_statistics": True,
            "launch_tensorboard": True,
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), normalize_targets=True, num_cls=80)],
            "metric_to_watch": "mAP@0.50:0.95",
        }
        trainer.train(model=model, training_params=training_params, train_loader=coco2017_train(), valid_loader=coco2017_val())


if __name__ == "__main__":
    unittest.main()
