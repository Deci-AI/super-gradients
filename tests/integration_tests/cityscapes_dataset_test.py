import unittest
from typing import Type

import pkg_resources
import yaml
from torch.utils.data import DataLoader, Dataset

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
    get,
)
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CityscapesDataset, CityscapesConcatDataset


class CityscapesDatasetTest(unittest.TestCase):
    def _cityscapes_dataset_params(self):
        default_config_path = pkg_resources.resource_filename("super_gradients.recipes", "dataset_params/cityscapes_dataset_params.yaml")
        with open(default_config_path, "r") as file:
            dataset_params = yaml.safe_load(file)
        return dataset_params

    def _cityscapes_al_dataset_params(self):
        default_config_path = pkg_resources.resource_filename("super_gradients.recipes", "dataset_params/cityscapes_al_dataset_params.yaml")
        with open(default_config_path, "r") as file:
            dataset_params = yaml.safe_load(file)
        return dataset_params

    def dataloader_tester(self, dl: DataLoader, dataset_cls: Type[Dataset] = CityscapesDataset):
        self.assertTrue(isinstance(dl, DataLoader))
        self.assertTrue(isinstance(dl.dataset, dataset_cls))
        it = iter(dl)
        for _ in range(10):
            next(it)

    def test_train_dataset_creation(self):
        dataset_params = self._cityscapes_dataset_params()
        train_dataset = CityscapesDataset(**dataset_params["train_dataset_params"])
        for i in range(10):
            image, mask = train_dataset[i]

    def test_al_train_dataset_creation(self):
        dataset_params = self._cityscapes_al_dataset_params()
        train_dataset = CityscapesConcatDataset(**dataset_params["train_dataset_params"])
        for i in range(10):
            image, mask = train_dataset[i]

    def test_val_dataset_creation(self):
        dataset_params = self._cityscapes_dataset_params()
        val_dataset = CityscapesDataset(**dataset_params["val_dataset_params"])
        for i in range(10):
            image, mask = val_dataset[i]

    def test_cityscapes_train_dataloader(self):
        dl_train = cityscapes_train()
        self.dataloader_tester(dl_train)

    def test_cityscapes_al_train_dataloader(self):
        dataset_params = self._cityscapes_al_dataset_params()
        # Same dataloader creation as in `train_from_recipe`
        dl_train = get(
            name=None,
            dataset_params=dataset_params["train_dataset_params"],
            dataloader_params=dataset_params["train_dataloader_params"],
        )
        self.dataloader_tester(dl_train, dataset_cls=CityscapesConcatDataset)

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
