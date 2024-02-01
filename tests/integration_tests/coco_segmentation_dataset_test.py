import unittest

import hydra
import pkg_resources
import yaml
from torch.utils.data import DataLoader

from super_gradients.training.dataloaders.dataloaders import coco_segmentation_train, coco_segmentation_val
from super_gradients.training.datasets.segmentation_datasets.coco_segmentation import CoCoSegmentationDataSet


class CocoSegmentationDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        default_config_path = pkg_resources.resource_filename("super_gradients.recipes", "dataset_params/coco_segmentation_dataset_params.yaml")
        with open(default_config_path, "r") as file:
            self.recipe = yaml.safe_load(file)

        self.recipe = hydra.utils.instantiate(self.recipe)

    def dataloader_tester(self, dl: DataLoader):
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, CoCoSegmentationDataSet))
        it = iter(dl)
        for _ in range(10):
            next(it)

    def test_train_dataset_creation(self):
        train_dataset = CoCoSegmentationDataSet(**self.recipe["train_dataset_params"])
        for i in range(10):
            image, mask = train_dataset[i]

    def test_val_dataset_creation(self):
        val_dataset = CoCoSegmentationDataSet(**self.recipe["val_dataset_params"])
        for i in range(10):
            image, mask = val_dataset[i]

    def test_coco_seg_train_dataloader(self):
        dl_train = coco_segmentation_train()
        self.dataloader_tester(dl_train)

    def test_coco_seg_val_dataloader(self):
        dl_val = coco_segmentation_val()
        self.dataloader_tester(dl_val)


if __name__ == "__main__":
    unittest.main()
