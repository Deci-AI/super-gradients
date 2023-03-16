import unittest
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val_deci_yolo
from super_gradients.training import Trainer

# - _target_: super_gradients.training.metrics.DetectionMetrics
# normalize_targets: True
# post_prediction_callback:
# _target_: super_gradients.training.models.detection_models.yolo_base.YoloPostPredictionCallback
# iou: 0.65
# conf: 0.01
# num_cls: 80
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models import YoloPostPredictionCallback


class MyTestCase(unittest.TestCase):
    def test_something(self):
        trainer = Trainer("test_deci_yolo_s")
        model = models.get("deciyolo_s", num_classes=80, pretrained_weights="coco")
        dl = coco2017_val_deci_yolo()
        metrics = [DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(iou=0.65, conf=0.01), num_cls=80, normalize_targets=False)]
        trainer.test(model=model, test_loader=dl, test_metrics_list=metrics)


if __name__ == "__main__":
    unittest.main()
