import unittest
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val_yolo_sg
from super_gradients.training import Trainer
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


class YoloSGIntegrationTest(unittest.TestCase):
    def test_yolo_sg_s_coco(self):
        trainer = Trainer("test_yolo_sg_s")
        model = models.get("yolo_sg_s", num_classes=80, pretrained_weights="coco")
        dl = coco2017_val_yolo_sg()
        metric = DetectionMetrics(
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
            num_cls=80,
        )
        metric_values = trainer.test(model=model, test_loader=dl, test_metrics_list=[metric])
        self.assertAlmostEqual(metric_values[metric.map_str], 0.475, delta=0.001)

    def test_yolo_sg_m_coco(self):
        trainer = Trainer("test_yolo_sg_m")
        model = models.get("yolo_sg_m", num_classes=80, pretrained_weights="coco")
        dl = coco2017_val_yolo_sg()
        metric = DetectionMetrics(
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
            num_cls=80,
        )
        metric_values = trainer.test(model=model, test_loader=dl, test_metrics_list=[metric])
        self.assertAlmostEqual(metric_values[metric.map_str], 0.5155, delta=0.001)

    def test_yolo_sg_l_coco(self):
        trainer = Trainer("test_yolo_sg_l")
        model = models.get("yolo_sg_l", num_classes=80, pretrained_weights="coco")
        dl = coco2017_val_yolo_sg()
        metric = DetectionMetrics(
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
            num_cls=80,
        )
        metric_values = trainer.test(model=model, test_loader=dl, test_metrics_list=[metric])
        self.assertAlmostEqual(metric_values[metric.map_str], 0.5222, delta=0.001)


if __name__ == "__main__":
    unittest.main()
