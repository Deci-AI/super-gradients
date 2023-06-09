import os.path
import unittest

import numpy as np
import torch

from super_gradients.common.object_names import Models
from super_gradients.module_interfaces import HasPredict
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_pose_val
from super_gradients.training.datasets.pose_estimation_datasets import DEKRTargetsGenerator


class TestPoseEstimationDataset(unittest.TestCase):
    def test_dekr_target_generator(self):
        target_generator = DEKRTargetsGenerator(
            output_stride=4,
            sigma=2,
            center_sigma=4,
            bg_weight=0.1,
            offset_radius=4,
        )

        joints = np.random.randint(0, 255, (4, 17, 3))
        joints[:, :, 2] = 1

        heatmaps, mask, offset_map, offset_weight = target_generator(
            image=torch.zeros((3, 256, 256)),
            joints=joints,
            mask=np.ones((256, 256)),
        )

        self.assertEqual(heatmaps.shape, (18, 64, 64))
        self.assertEqual(mask.shape, (18, 64, 64))
        self.assertEqual(offset_map.shape, (34, 64, 64))
        self.assertEqual(offset_weight.shape, (34, 64, 64))

    def test_get_dataset_preprocessing_params(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "coco2017")

        loader = coco2017_pose_val(dataset_params={"target_generator": None, "data_dir": data_dir, "json_file": "annotations/person_keypoints_val2017.json"})
        preprocessing_params = loader.dataset.get_dataset_preprocessing_params()
        self.assertIsNotNone(preprocessing_params)

        dekr: HasPredict = models.get(Models.DEKR_W32_NO_DC, pretrained_weights="coco_pose")
        dekr.set_dataset_processing_params(**preprocessing_params)
        dekr.predict(np.zeros((640, 640, 3), dtype=np.uint8))
