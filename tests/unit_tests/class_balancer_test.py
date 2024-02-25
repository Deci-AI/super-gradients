import os
import tempfile

import numpy as np
import torch
import unittest

from torch.utils.data import Dataset

from super_gradients.dataset_interfaces import HasClassesInformation
from super_gradients.training.datasets.samplers.class_balanced_sampler import ClassBalancer


class SingleLabelUnbalancedDataset(Dataset, HasClassesInformation):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_empty_annotations = False

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx] * idx)  # no class 0

    def get_sample_classes_information(self, index) -> np.ndarray:
        info = np.zeros(self.num_classes, dtype=np.int)
        info[index] = index
        return info

    def get_dataset_classes_information(self) -> np.ndarray:
        return np.diag(np.arange(self.num_classes))


class MultiLabelUnbalancedDataset(Dataset, HasClassesInformation):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_empty_annotations = False

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx, 0])  # class 0 appears everywhere, other classes appear only once.

    def get_sample_classes_information(self, index) -> np.ndarray:
        info = np.zeros(self.num_classes, dtype=int)
        info[index] = 1
        info[0] += 1
        return info

    def get_dataset_classes_information(self) -> np.ndarray:
        diag = np.eye(self.num_classes, dtype=int)
        diag[:, 0] += 1
        return diag


class ClassBalancerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.single_label_dataset = SingleLabelUnbalancedDataset(num_classes=5)  # [[], [1], [2,2], [3,3,3], [4,4,4,4]]
        self.multi_label_dataset = MultiLabelUnbalancedDataset(num_classes=5)  # [[0,0], [1,0], [2,0], [3,0], [4,0]]

    def test_without_oversampling(self):
        repeat_factors = ClassBalancer.get_sample_repeat_factors(
            self.single_label_dataset,
            oversample_threshold=0.0,
        )
        expected_mappings = [1] * len(self.single_label_dataset)
        self.assertListEqual(expected_mappings, repeat_factors)

    def test_oversampling_frequent_classes_less_often_than_scarce(self):
        repeat_factors = ClassBalancer.get_sample_repeat_factors(
            self.single_label_dataset,
            oversample_threshold=1.0,
        )

        # reminder: samples = [[], [1], [2,2], [3,3,3], [4,4,4,4]]
        self.assertEqual(repeat_factors[0], 1.0)  # do not over sample empty annotations

        # expected something like [1.0, a, b, c, d], a>b>c>d>1.0
        diffs = np.diff(repeat_factors[1:])
        self.assertTrue(np.all(diffs < 0.0))

    def test_multi_class_over_sampling(self):
        """
        Interestingly, when we have a class that appears in every sample ([[0,0], [1,0], [2,0], [3,0], [4,0]]),
        and other samples have the same frequencies, we are still oversampling samples, but use the same repeat factor for all.
        The reason is that originally we have #0 class appearing 6 times, and other classes appear 1 time, which is 6x freq; after resampling,
        we have #0 class appearing ~14 times, and other classes appear ~3 times. Note that lower bound for class #0 is 4x freq, and after resampling it is 4.6x.
        """
        repeat_factors = ClassBalancer.get_sample_repeat_factors(
            self.multi_label_dataset,
            oversample_threshold=1.0,
        )

        # reminder: samples = [[0,0], [1,0], [2,0], [3,0], [4,0]]
        self.assertEqual(1.0, repeat_factors[0])  # do not over sample the biggest class

        # expected something like [1.0, a, b, c, d], a=b=c=d>x>1.0
        diffs = np.diff(repeat_factors[1:])
        self.assertTrue(np.all(diffs == 0.0))

    def test_no_oversample_below_threshold(self):
        repeat_factors = ClassBalancer.get_sample_repeat_factors(
            self.single_label_dataset,
            oversample_threshold=0.5,
        )

        # reminder: samples = [[], [1], [2,2], [3,3,3], [4,4,4,4]]
        # overall we have 5 images, class #1 appears 1/5 (in image 1), #2 appears 2/5 (image 2), #3 appears 3/5 (image 3), #4 appears 4/5 (image 4).
        # We will not oversample IMAGES 3 and 4, nor the empty image 0.
        oversampled_indices = np.array([False, True, True, False, False])
        self.assertTrue(np.all(np.array(repeat_factors)[~oversampled_indices] == 1.0))  # all

        # make sure indices that are oversampled are with expected repeat factor
        self.assertTrue(np.all(np.diff(np.array(repeat_factors)[oversampled_indices]) < 0.0))

    def test_precomputed_repeat_factors(self):
        repeat_factors = ClassBalancer.get_sample_repeat_factors(
            self.single_label_dataset,
            oversample_threshold=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            precomputed_file = os.path.join(temp_dir, "precomputed_repeat_factors.json")
            ClassBalancer.precompute_sample_repeat_factors(precomputed_file, self.single_label_dataset)
            loaded_repeat_factors = ClassBalancer.from_precomputed_sample_repeat_factors(precomputed_file)

        np.testing.assert_almost_equal(repeat_factors, loaded_repeat_factors, decimal=3)


if __name__ == "__main__":
    unittest.main()
