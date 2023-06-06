import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset

from super_gradients.training import dataloaders


class TestMinSamplesSingleNode(unittest.TestCase):
    def test_min_samples(self):
        dataset_size = 64
        image_size = 32
        images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
        ground_truth = torch.LongTensor(np.zeros((dataset_size)))
        dataloader = dataloaders.get(dataset=TensorDataset(images, ground_truth), dataloader_params={"batch_size": 4, "min_samples": 80, "drop_last": True})
        self.assertEqual(len(dataloader), 80 / 4)


if __name__ == "__main__":
    unittest.main()
