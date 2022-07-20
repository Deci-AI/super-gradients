import os
import unittest

from super_gradients.training import Trainer, utils as core_utils
from super_gradients.training.datasets import CoCoDetectionDatasetInterface
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.models.detection_models.yolov5_base import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import base_detection_collate_fn, DetectionVisualization


class TestDetectionUtils(unittest.TestCase):
    def test_visualization(self):
        # Create dataset
        dataset_params = {"batch_size": 4, "val_batch_size": 4, "train_image_size": 320, "val_image_size": 320,
                          "val_collate_fn": base_detection_collate_fn,
                          "train_collate_fn": base_detection_collate_fn,
                          "val_sample_loading_method": "default"
                          }
        dataset = CoCoDetectionDatasetInterface(dataset_params)

        # Create Yolo model
        trainer = Trainer('visualization_test',
                        model_checkpoints_location='local',
                        post_prediction_callback=YoloV5PostPredictionCallback())
        trainer.connect_dataset_interface(dataset, data_loader_num_workers=8)
        trainer.build_model("yolo_v5s")

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
        for i in range(dataset_params['val_batch_size']):
            img_path = img_name.format(trainer.checkpoints_dir_path, batch_i, i)
            self.assertTrue(os.path.exists(img_path))
            os.remove(img_path)


if __name__ == '__main__':
    unittest.main()
