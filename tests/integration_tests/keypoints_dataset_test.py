import os
import unittest

import pkg_resources
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from super_gradients.training.dataloaders.dataloaders import _process_dataset_params
from super_gradients.training.datasets.pose_estimation_datasets import COCOKeypointsDataset
from super_gradients.training.utils.hydra_utils import normalize_path


class KeypointsDatasetIntegrationTest(unittest.TestCase):
    def test_keypoints_dataset_instantiation(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        dataset_config = os.path.join("dataset_params", "coco_dekr_pose_estimation_dataset_params")
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            # config is relative to a module
            cfg = compose(config_name=normalize_path(dataset_config))
            dataset_params = _process_dataset_params(cfg, dict(), True)

        dataset = COCOKeypointsDataset(**dataset_params)
        self.assertIsNotNone(dataset.__getitem__(0))


if __name__ == "__main__":
    unittest.main()
