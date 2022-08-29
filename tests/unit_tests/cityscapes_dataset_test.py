import unittest

import pkg_resources
import yaml

from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CityscapesDataset


class CityscapesDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        default_config_path = pkg_resources.resource_filename("super_gradients.recipes",
                                                              "dataset_params/cityscapes_dataset_params.yaml")
        with open(default_config_path, 'r') as file:
            self.recipe = yaml.safe_load(file)

    def test_train_dataset_creation(self):
        train_dataset = CityscapesDataset(**self.recipe['train_dataset_params'])
        for i in range(10):
            image, mask = train_dataset[i]


if __name__ == '__main__':
    unittest.main()
