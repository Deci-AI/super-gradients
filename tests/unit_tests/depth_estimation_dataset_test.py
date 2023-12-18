import unittest
from pathlib import Path
import os

import numpy as np

from super_gradients.training.datasets.depth_estimation_datasets import NYUv2DepthEstimationDataset


class DepthEstimationDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mini_nyuv2_data_dir = str(Path(__file__).parent.parent / "data" / "nyu2_mini_test")
        self.mini_nyuv2_df_path = os.path.join(self.mini_nyuv2_data_dir, "nyu2_mini_test.csv")

    def test_normal_nyuv2_creation(self):
        dataset = NYUv2DepthEstimationDataset(root=self.mini_nyuv2_data_dir, df_path=self.mini_nyuv2_df_path)
        for i, (image, depth_map) in enumerate(dataset):
            self.assertTrue(isinstance(image, np.ndarray))
            self.assertTrue(isinstance(depth_map, np.ndarray))
        self.assertTrue(len(dataset) == 10 and i == 9)

    def test_depth_estimation_plot(self):
        dataset = NYUv2DepthEstimationDataset(root=self.mini_nyuv2_data_dir, df_path=self.mini_nyuv2_df_path)
        dataset.plot(max_samples_per_plot=8)


if __name__ == "__main__":
    unittest.main()
