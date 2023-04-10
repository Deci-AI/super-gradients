import unittest
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val_deci_yolo
from super_gradients.training import Trainer
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


class DeciYoloTest(unittest.TestCase):
    def test_deciyolo_s_coco(self):
        trainer = Trainer("test_deci_yolo_s")
        model = models.get("deciyolo_s", num_classes=80, pretrained_weights="coco")
        dl = coco2017_val_deci_yolo()
        metrics = [
            DetectionMetrics(
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
                num_cls=80,
            )
        ]
        map_result = trainer.test(model=model, test_loader=dl, test_metrics_list=metrics)[2]
        self.assertAlmostEqual(map_result, 0.475, delta=0.001)

    def test_deciyolo_m_coco(self):
        trainer = Trainer("test_deci_yolo_m")
        model = models.get("deciyolo_m", num_classes=80, pretrained_weights="coco")
        dl = coco2017_val_deci_yolo()
        metrics = [
            DetectionMetrics(
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
                num_cls=80,
            )
        ]
        map_result = trainer.test(model=model, test_loader=dl, test_metrics_list=metrics)[2]
        self.assertAlmostEqual(map_result, 0.5155, delta=0.001)

    def test_deciyolo_l_coco(self):
        trainer = Trainer("test_deci_yolo_l")
        model = models.get("deciyolo_l", num_classes=80, pretrained_weights="coco")
        dl = coco2017_val_deci_yolo()
        metrics = [
            DetectionMetrics(
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
                num_cls=80,
            )
        ]
        map_result = trainer.test(model=model, test_loader=dl, test_metrics_list=metrics)[2]
        self.assertAlmostEqual(map_result, 0.5222, delta=0.001)


if __name__ == "__main__":
    unittest.main()
