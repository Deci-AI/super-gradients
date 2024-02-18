import numpy as np
import torch
import unittest

from torch.utils.data import Dataset

from super_gradients.training.datasets.balancing_classes_utils import get_repeat_factors


class SingleLabelUnbalancedDataset(Dataset):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_empty_annotations = False

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx] * idx)  # no class 0


class MultiLabelUnbalancedDataset(Dataset):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_empty_annotations = False

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx, 0])  # class 0 appears everywhere, other classes appear only once.


class ClassBalancingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.single_label_dataset = SingleLabelUnbalancedDataset(num_classes=5)  # [[], [1], [2,2], [3,3,3], [4,4,4,4]]
        self.multi_label_dataset = MultiLabelUnbalancedDataset(num_classes=5)  # [[0,0], [1,0], [2,0], [3,0], [4,0]]

    def test_without_oversampling(self):
        repeat_factors = get_repeat_factors(
            index_to_classes=lambda idx: self.single_label_dataset[idx].tolist(),
            num_classes=self.single_label_dataset.num_classes,
            dataset_length=len(self.single_label_dataset),
            ignore_empty_annotations=self.single_label_dataset.ignore_empty_annotations,
            oversample_threshold=0.0,
        )
        expected_mappings = [1.0] * len(self.single_label_dataset)
        self.assertListEqual(expected_mappings, repeat_factors)

    def test_oversampling_frequent_classes_less_often_than_scarce(self):
        repeat_factors = get_repeat_factors(
            index_to_classes=lambda idx: self.single_label_dataset[idx].tolist(),
            num_classes=self.single_label_dataset.num_classes,
            dataset_length=len(self.single_label_dataset),
            ignore_empty_annotations=self.single_label_dataset.ignore_empty_annotations,
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
        we have #0 class appearing 14 times, and other classes appear 3 times. Note that lower bound for class #0 is 4x freq, and after resampling it is 4.6x.
        """
        repeat_factors = get_repeat_factors(
            index_to_classes=lambda idx: self.multi_label_dataset[idx].tolist(),
            num_classes=self.multi_label_dataset.num_classes,
            dataset_length=len(self.multi_label_dataset),
            ignore_empty_annotations=self.multi_label_dataset.ignore_empty_annotations,
            oversample_threshold=1.0,
        )

        # reminder: samples = [[0,0], [1,0], [2,0], [3,0], [4,0]]
        self.assertEqual(repeat_factors[0], 1.0)  # do not over sample the biggest class

        # expected something like [1.0, a, b, c, d], a=b=c=d>1.0
        diffs = np.diff(repeat_factors[1:])
        self.assertTrue(np.all(diffs == 0.0))

    def test_no_oversample_below_threshold(self):
        repeat_factors = get_repeat_factors(
            index_to_classes=lambda idx: self.single_label_dataset[idx].tolist(),
            num_classes=self.single_label_dataset.num_classes,
            dataset_length=len(self.single_label_dataset),
            ignore_empty_annotations=self.single_label_dataset.ignore_empty_annotations,
            oversample_threshold=0.5,
        )

        # reminder: samples = [[], [1], [2,2], [3,3,3], [4,4,4,4]]
        # overall we have 5 images, class #1 appears 1/5 (in image 1), #2 appears 2/5 (image 2), #3 appears 3/5 (image 3), #4 appears 4/5 (image 4).
        # We will not oversample IMAGES 3 and 4, nor the empty image 0.
        oversampled_indices = np.array([False, True, True, False, False])
        self.assertTrue(np.all(np.array(repeat_factors)[~oversampled_indices] == 1.0))  # all

        # make sure indices that are oversampled are with expected repeat factor
        self.assertTrue(np.all(np.diff(np.array(repeat_factors)[oversampled_indices]) < 0.0))


if __name__ == "__main__":
    unittest.main()
