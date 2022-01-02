import os
import unittest

from super_gradients.training import SgModel, utils as core_utils
from super_gradients.training.datasets import CoCoDetectionDatasetInterface
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.models.detection_models.yolov5 import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import base_detection_collate_fn, DetectionVisualization
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import PascalVOC2UnifiedDetectionDataSetInterface

class TestDetectionUtils(unittest.TestCase):
    def test_visualization(self):
        # Create dataset
        PASCAL_VOC_2012_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor'
        ]
        dataset_params = {"batch_size": 48,
                          "val_batch_size": 48,
                          "train_image_size": 512,
                          "val_image_size": 512,
                          "val_collate_fn": base_detection_collate_fn,
                          "train_collate_fn": base_detection_collate_fn,
                          "train_sample_loading_method": "mosaic",
                          "val_sample_loading_method": "default",
                          "dataset_hyper_param": {
                              "hsv_h": 0.0138,  # IMAGE HSV-Hue AUGMENTATION (fraction)
                              "hsv_s": 0.664,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
                              "hsv_v": 0.464,  # IMAGE HSV-Value AUGMENTATION (fraction)
                              "degrees": 0.373,  # IMAGE ROTATION (+/- deg)
                              "translate": 0.245,  # IMAGE TRANSLATION (+/- fraction)
                              "scale": 0.898,  # IMAGE SCALE (+/- gain)
                              "shear": 0.602,
                              "mixup": 0.243
                          }  # IMAGE SHEAR (+/- deg)
                          }

        model = SgModel("yolov5m_pascal_finetune_viz_2007",
                        post_prediction_callback=YoloV5PostPredictionCallback())

        dataset = PascalVOC2UnifiedDetectionDataSetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset, data_loader_num_workers=20)
        model.build_model("yolo_v5m", arch_params={"pretrained_weights": "coco"})

        post_prediction_callback = YoloV5PostPredictionCallback()

        valid_loader = model.valid_loader
        batch_i, (imgs, targets) = 0, next(iter(valid_loader))
        imgs = core_utils.tensor_container_to_device(imgs, model.device)
        targets = core_utils.tensor_container_to_device(targets, model.device)
        output = model.net(imgs)
        output = model.post_prediction_callback(output)
        # Visualize the batch
        DetectionVisualization.visualize_batch(imgs, output, targets, batch_i,
                                               PASCAL_VOC_2012_CLASSES, model.checkpoints_dir_path)

        # Assert images ware created and delete them
        img_name = '{}/{}_{}.jpg'
        for i in range(dataset_params['val_batch_size']):
            img_path = img_name.format(model.checkpoints_dir_path, batch_i, i)
            self.assertTrue(os.path.exists(img_path))
            # os.remove(img_path)


if __name__ == '__main__':
    unittest.main()
