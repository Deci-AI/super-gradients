import torch
import unittest

from torch.utils.data import Dataset

from super_gradients.training.datasets.balancing_classes_utils import IndexMappingDatasetWrapper


class DummyDataset(Dataset):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_empty_annotations = False

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(idx)


class DatasetIndexMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_dataset = DummyDataset(num_classes=5)

    def test_mapping_indices_that_does_nothing(self):
        wrapper = IndexMappingDatasetWrapper(self.dummy_dataset, list(range(len(self.dummy_dataset))))
        self.assertEqual(len(wrapper), len(self.dummy_dataset))
        for i in range(len(wrapper)):
            self.assertEqual(self.dummy_dataset[i], wrapper[i])

    def test_mapping_indices_that_samples_only_specific_index(self):
        c = 3
        i = 1
        mapping = [i] * c
        wrapper = IndexMappingDatasetWrapper(self.dummy_dataset, mapping)
        for j in range(len(wrapper)):
            self.assertEqual(self.dummy_dataset[i], wrapper[j])


if __name__ == "__main__":
    unittest.main()
