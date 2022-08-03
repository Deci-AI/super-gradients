import unittest

from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CocoDetectionDatasetInterfaceV2
from super_gradients.training.metrics.detection_metrics import DetectionMetrics

from super_gradients.training import SgModel
from super_gradients.training.datasets import CoCoDetectionDatasetInterface
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN, DetectionCollateFN, \
    DetectionTargetsFormat


class TestDatasetStatisticsTensorboardLogger(unittest.TestCase):

    def test_dataset_statistics_tensorboard_logger(self):
        """
        ** IMPORTANT NOTE **
        This test is not the usual fail/pass test - it is a visual test. The success criteria is your own visual check
        After launching the test, follow the log the see where was the tensorboard opened. open the tensorboard in your
        browser and make sure the text and plots in the tensorboard are as expected.
        """
        # Create dataset
        dataset = CocoDetectionDatasetInterfaceV2(dataset_params={"data_dir": "/data/coco",
                                                                  "train_subdir": "images/train2017",
                                                                  "val_subdir": "images/val2017",
                                                                  "train_json_file": "instances_train2017.json",
                                                                  "val_json_file": "instances_val2017.json",
                                                                  "batch_size": 16,
                                                                  "val_batch_size": 128,
                                                                  "val_image_size": 640,
                                                                  "train_image_size": 640,
                                                                  "hgain": 5,
                                                                  "sgain": 30,
                                                                  "vgain": 30,
                                                                  "mixup_prob": 1.0,
                                                                  "degrees": 10.,
                                                                  "shear": 2.0,
                                                                  "flip_prob": 0.5,
                                                                  "hsv_prob": 1.0,
                                                                  "mosaic_scale": [0.1, 2],
                                                                  "mixup_scale": [0.5, 1.5],
                                                                  "mosaic_prob": 1.,
                                                                  "translate": 0.1,
                                                                  "val_collate_fn": CrowdDetectionCollateFN(),
                                                                  "train_collate_fn": DetectionCollateFN(),
                                                                  "cache_dir_path": None,
                                                                  "cache_train_images": False,
                                                                  "cache_val_images": False,
                                                                  "targets_format": DetectionTargetsFormat.LABEL_CXCYWH,
                                                                  "with_crowd": True,
                                                                  "filter_box_candidates": False,
                                                                  "wh_thr": 0,
                                                                  "ar_thr": 0,
                                                                  "area_thr": 0
                                                                  })

        model = SgModel('dataset_statistics_visual_test',
                        model_checkpoints_location='local',
                        post_prediction_callback=YoloPostPredictionCallback())
        model.connect_dataset_interface(dataset, data_loader_num_workers=8)
        model.build_model("yolox_s")

        training_params = {"max_epochs": 1,  # we dont really need the actual training to run
                           "lr_mode": "cosine",
                           "initial_lr": 0.01,
                           "loss": "yolox_loss",
                           "criterion_params": {"strides": [8, 16, 32], "num_cls": 80},
                           "dataset_statistics": True,
                           "launch_tensorboard": True,
                           "valid_metrics_list": [
                               DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(),
                                                normalize_targets=True,
                                                num_cls=80)],

                           "loss_logging_items_names": ["iou", "obj", "cls", "l1", "num_fg", "Loss"],
                           "metric_to_watch": "mAP@0.50:0.95",
                           }
        model.train(training_params=training_params)


if __name__ == '__main__':
    unittest.main()
