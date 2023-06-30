import unittest
from pprint import pprint

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val_yolox
from super_gradients.training import Trainer
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models import YoloPostPredictionCallback
from super_gradients.training.utils.detection_utils import NMS_Type


class YoloXPostProcessingTest(unittest.TestCase):
    """ """

    def setUp(self) -> None:
        # confidence 0.001, IoU threshold 0.6
        self.default_post_prediction_params = dict(conf=0.03, iou=0.65, max_predictions=300)

    def test_yolox_l_matrix_nms_multi_label_per_box_True(self):
        post_prediction_params = dict(**self.default_post_prediction_params, nms_type=NMS_Type.MATRIX, class_agnostic_nms=False)

        self._evaluate_yolox(post_prediction_params)

    def test_yolox_l_matrix_nms_class_agnostic_nms_True(self):
        post_prediction_params = dict(**self.default_post_prediction_params, nms_type=NMS_Type.MATRIX, class_agnostic_nms=True)

        self._evaluate_yolox(post_prediction_params)

    def test_yolox_l_iterative_nms_multi_label_per_box_True_class_agnostic_nms_False(self):
        post_prediction_params = dict(**self.default_post_prediction_params, nms_type=NMS_Type.ITERATIVE, multi_label_per_box=True, class_agnostic_nms=False)

        self._evaluate_yolox(post_prediction_params)

    def test_yolox_l_iterative_nms_multi_label_per_box_False_class_agnostic_nms_False(self):
        post_prediction_params = dict(**self.default_post_prediction_params, nms_type=NMS_Type.ITERATIVE, multi_label_per_box=False, class_agnostic_nms=False)

        self._evaluate_yolox(post_prediction_params)

    def test_yolox_l_iterative_nms_multi_label_per_box_True_class_agnostic_nms_True(self):
        post_prediction_params = dict(**self.default_post_prediction_params, nms_type=NMS_Type.ITERATIVE, multi_label_per_box=True, class_agnostic_nms=True)

        self._evaluate_yolox(post_prediction_params)

    def test_yolox_l_iterative_nms_multi_label_per_box_False_class_agnostic_nms_True(self):
        post_prediction_params = dict(**self.default_post_prediction_params, nms_type=NMS_Type.ITERATIVE, multi_label_per_box=False, class_agnostic_nms=True)

        self._evaluate_yolox(post_prediction_params)

    def _evaluate_yolox(self, post_prediction_params):
        trainer = Trainer("test_yolox_l")
        model = models.get(Models.YOLOX_L, num_classes=80, pretrained_weights="coco")
        val_loader = coco2017_val_yolox(dataset_params=dict(data_dir="e:/coco2017"), dataloader_params=dict(num_workers=0))

        metric = DetectionMetrics(
            normalize_targets=True,
            post_prediction_callback=YoloPostPredictionCallback(**post_prediction_params),
            num_cls=80,
        )
        metric_values = trainer.test(model=model, test_loader=val_loader, test_metrics_list=[metric])

        print()
        pprint(post_prediction_params)
        pprint(metric_values)
        print()


if __name__ == "__main__":
    unittest.main()
