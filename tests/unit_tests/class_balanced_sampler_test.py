import random
from typing import Dict

import numpy as np
import torch
import unittest

from torch.utils.data import Dataset

from super_gradients.dataset_interfaces import HasClassesInformation
from super_gradients.training import dataloaders
from super_gradients.training.datasets.samplers.class_balanced_sampler import ClassBalancedSampler


class DummyFreqDataset(Dataset, HasClassesInformation):
    def __init__(self, class_id_to_frequency: Dict[int, int], total_samples: int) -> None:
        self.total_samples = total_samples
        self.num_classes = len(class_id_to_frequency)
        self.class_id_to_frequency = class_id_to_frequency
        self.ignore_empty_annotations = True
        self._setup_data_source()
        super().__init__()

    def _setup_data_source(self) -> int:
        flattened_list = list()
        for k, v in self.class_id_to_frequency.items():
            flattened_list.extend([k] * v)

        random.shuffle(flattened_list)

        self.idx_to_classes = np.array_split(flattened_list, self.total_samples)
        return len(self.idx_to_classes)

    def __len__(self) -> int:
        return len(self.idx_to_classes)

    def __getitem__(self, index: int):
        return self.idx_to_classes[index]

    def get_sample_classes_information(self, index: int) -> np.ndarray:
        classes = self.idx_to_classes[index]
        return np.bincount(classes, minlength=self.num_classes)

    def get_dataset_classes_information(self) -> np.ndarray:
        return np.vstack([self.get_sample_classes_information(index) for index in range(len(self))])


class ClassBalancedSamplerTest(unittest.TestCase):
    def test_balancing_classes_that_are_with_same_frequency(self):
        id_to_freq = {0: 30000, 1: 30000, 2: 30000}
        total_samples = 60000
        dataset = DummyFreqDataset(class_id_to_frequency=id_to_freq, total_samples=total_samples)
        sampler = ClassBalancedSampler(dataset=dataset)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, sampler=sampler)

        classes_sampled = {k: 0 for k in id_to_freq.keys()}

        for batch in dataloader:
            for element in batch:
                for cls in element:
                    classes_sampled[cls.item()] += 1

        for k in classes_sampled.keys():
            expected_freq = id_to_freq[k] / total_samples
            sampled_freq = classes_sampled[k] / total_samples
            self.assertAlmostEqual(expected_freq, sampled_freq, places=1)

    def test_balancing_scarce_classes(self):
        id_to_freq = {0: 10000, 1: 1000, 2: 10000}
        total_samples = 15000
        dataset = DummyFreqDataset(class_id_to_frequency=id_to_freq, total_samples=total_samples)
        sampler = ClassBalancedSampler(dataset=dataset)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, sampler=sampler)

        classes_sampled = {k: 0 for k in id_to_freq.keys()}

        for batch in dataloader:
            for element in batch:
                for cls in element:
                    classes_sampled[cls.item()] += 1

        for k in classes_sampled.keys():
            original_freq = id_to_freq[k] / total_samples
            sampled_freq = classes_sampled[k] / total_samples
            if k == 1:  # over sampled class
                self.assertGreater(sampled_freq, original_freq)
            else:
                self.assertLess(sampled_freq, original_freq)

    def test_get_from_config(self):
        id_to_freq = {0: 10, 1: 1, 2: 10}
        total_samples = 15
        dataset = DummyFreqDataset(class_id_to_frequency=id_to_freq, total_samples=total_samples)

        dataloader_params = {
            "batch_size": 4,
            "sampler": {"ClassBalancedSampler": {"oversample_threshold": 1.0, "oversample_aggressiveness": 1.5}},
            "drop_last": True,
        }

        dataloader = dataloaders.get(dataset=dataset, dataloader_params=dataloader_params)
        self.assertTrue(isinstance(dataloader.sampler, ClassBalancedSampler))


if __name__ == "__main__":
    unittest.main()
