import unittest

import pkg_resources
import yaml
from torch.utils.data import DataLoader

from super_gradients.training.dataloaders.dataloaders import (
    cityscapes_train,
    cityscapes_val,
    cityscapes_stdc_seg50_train,
    cityscapes_stdc_seg50_val,
    cityscapes_stdc_seg75_val,
    cityscapes_ddrnet_train,
    cityscapes_regseg48_val,
    cityscapes_regseg48_train,
    cityscapes_ddrnet_val,
    cityscapes_stdc_seg75_train,
)
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CityscapesDataset


class CityscapesDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        default_config_path = pkg_resources.resource_filename("super_gradients.recipes", "dataset_params/cityscapes_dataset_params.yaml")
        with open(default_config_path, "r") as file:
            self.recipe = yaml.safe_load(file)

    def dataloader_tester(self, dl: DataLoader):
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, CityscapesDataset))
        it = iter(dl)
        for _ in range(10):
            next(it)

    def test_train_dataset_creation(self):
        train_dataset = CityscapesDataset(**self.recipe["train_dataset_params"])
        for i in range(10):
            image, mask = train_dataset[i]

    def test_val_dataset_creation(self):
        val_dataset = CityscapesDataset(**self.recipe["val_dataset_params"])
        for i in range(10):
            image, mask = val_dataset[i]

    def test_cityscapes_train_dataloader(self):
        dl_train = cityscapes_train()
        self.dataloader_tester(dl_train)

    def test_cityscapes_val_dataloader(self):
        dl_val = cityscapes_val()
        self.dataloader_tester(dl_val)

    def test_cityscapes_stdc_seg50_train_dataloader(self):
        dl_train = cityscapes_stdc_seg50_train()
        self.dataloader_tester(dl_train)

    def test_cityscapes_stdc_seg50_val_dataloader(self):
        dl_val = cityscapes_stdc_seg50_val()
        self.dataloader_tester(dl_val)

    def test_cityscapes_stdc_seg75_train_dataloader(self):
        dl_train = cityscapes_stdc_seg75_train()
        self.dataloader_tester(dl_train)

    def test_cityscapes_stdc_seg75_val_dataloader(self):
        dl_val = cityscapes_stdc_seg75_val()
        self.dataloader_tester(dl_val)

    def test_cityscapes_regseg48_train_dataloader(self):
        dl_train = cityscapes_regseg48_train()
        self.dataloader_tester(dl_train)

    def test_cityscapes_regseg48_val_dataloader(self):
        dl_val = cityscapes_regseg48_val()
        self.dataloader_tester(dl_val)

    def test_cityscapes_ddrnet_train_dataloader(self):
        dl_train = cityscapes_ddrnet_train()
        self.dataloader_tester(dl_train)

    def test_cityscapes_ddrnet_val_dataloader(self):
        dl_val = cityscapes_ddrnet_val()
        self.dataloader_tester(dl_val)


if __name__ == "__main__":
    unittest.main()
