import unittest
from pathlib import Path

import numpy as np

from super_gradients.training.datasets.optical_flow_datasets.kitti_dataset import KITTIOpticalFlowDataset


class OpticalFlowDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kitti_2015_data_dir = str(Path(__file__).parent.parent / "data" / "kitti_2015")

    def test_kitti_creation(self):
        dataset = KITTIOpticalFlowDataset(root=self.kitti_2015_data_dir)
        for i, (images, target) in enumerate(dataset):
            flow, valid = target
            self.assertTrue(isinstance(images, np.ndarray))
            self.assertTrue(isinstance(flow, np.ndarray))
            self.assertTrue(isinstance(valid, np.ndarray))
        self.assertTrue(len(dataset) == 10 and i == 9)

    def test_optical_flow_plot(self):
        dataset = KITTIOpticalFlowDataset(root=self.kitti_2015_data_dir)
        dataset.plot(max_samples_per_plot=8)


if __name__ == "__main__":
    unittest.main()
