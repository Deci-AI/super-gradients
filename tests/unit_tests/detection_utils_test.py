import os
import unittest

from super_gradients.training import Trainer, utils as core_utils
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CoCoDetectionDatasetInterface
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionVisualization, DetectionCollateFN, DetectionTargetsFormat


class TestDetectionUtils(unittest.TestCase):
    def test_visualization(self):
        # Create dataset
        dataset = CoCoDetectionDatasetInterface(dataset_params={"data_dir": "/data/coco",
                                                                "train_subdir": "images/train2017",
                                                                "val_subdir": "images/val2017",
                                                                "train_json_file": "instances_train2017.json",
                                                                "val_json_file": "instances_val2017.json",
                                                                "batch_size": 16,
                                                                "val_batch_size": 4,
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
                                                                "val_collate_fn": DetectionCollateFN(),
                                                                "train_collate_fn": DetectionCollateFN(),
                                                                "cache_dir_path": None,
                                                                "cache_train_images": False,
                                                                "cache_val_images": False,
                                                                "targets_format": DetectionTargetsFormat.LABEL_NORMALIZED_CXCYWH,
                                                                "with_crowd": False,
                                                                "filter_box_candidates": False,
                                                                "wh_thr": 0,
                                                                "ar_thr": 0,
                                                                "area_thr": 0
                                                                })

        # Create Yolo model
        trainer = Trainer('visualization_test',
                        model_checkpoints_location='local',
                        post_prediction_callback=YoloPostPredictionCallback())
        trainer.connect_dataset_interface(dataset, data_loader_num_workers=8)
        trainer.build_model("yolox_n", checkpoint_params={"pretrained_weights": "coco"})

        # Simulate one iteration of validation subset
        valid_loader = trainer.valid_loader
        batch_i, (imgs, targets) = 0, next(iter(valid_loader))
        imgs = core_utils.tensor_container_to_device(imgs, trainer.device)
        targets = core_utils.tensor_container_to_device(targets, trainer.device)
        output = trainer.net(imgs)
        output = trainer.post_prediction_callback(output)
        # Visualize the batch
        DetectionVisualization.visualize_batch(imgs, output, targets, batch_i,
                                               COCO_DETECTION_CLASSES_LIST, trainer.checkpoints_dir_path)

        # Assert images ware created and delete them
        img_name = '{}/{}_{}.jpg'
        for i in range(4):
            img_path = img_name.format(trainer.checkpoints_dir_path, batch_i, i)
            self.assertTrue(os.path.exists(img_path))
            os.remove(img_path)


if __name__ == '__main__':
    unittest.main()
