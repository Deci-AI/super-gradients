import unittest
import os
import shutil
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationDatasetInterface


class TestDataset(unittest.TestCase):

    def test_donwload_dataset(self):
        default_dataset_params = {"dataset_dir": os.path.expanduser("~/test_data/"),
                                  "s3_link": "s3://research-data1/data.zip"}

        dataset = ClassificationDatasetInterface(dataset_params=default_dataset_params)

        test_sample = dataset.get_test_sample()
        self.assertListEqual([3, 64, 64], list(test_sample[0].shape))
        shutil.rmtree(default_dataset_params["dataset_dir"])


if __name__ == '__main__':
    unittest.main()
