import os
import unittest

import pkg_resources
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

import super_gradients
from super_gradients.common.environment.path_utils import normalize_path
from super_gradients.training.dataloaders.dataloaders import _process_dataset_params
from super_gradients.training.datasets.pose_estimation_datasets import COCOKeypointsDataset


class PoseEstimationDatasetIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()

    def test_datasets_instantiation(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
        dataset_config = os.path.join("dataset_params", "coco_pose_estimation_dekr_dataset_params")
        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            # config is relative to a module
            cfg = compose(config_name=normalize_path(dataset_config))
            train_dataset_params = _process_dataset_params(cfg, dict(), True)
            val_dataset_params = _process_dataset_params(cfg, dict(), True)

            train_dataset = COCOKeypointsDataset(**train_dataset_params)
            assert train_dataset[0] is not None

            val_dataset = COCOKeypointsDataset(**val_dataset_params)
            assert val_dataset[0] is not None


if __name__ == "__main__":
    unittest.main()
