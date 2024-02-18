import random
from typing import Dict, Union, Any

import numpy as np
import torch
import unittest

from torch.utils.data import Dataset

from super_gradients.training.datasets.detection_datasets.detection_dataset_class_balancing_wrapper import DetectionClassBalancedDistributedSampler


class DummyDetectionDataset(Dataset):  # NOTE: we implement the needed stuff from DetectionDataset, but we do not inherit it because the ctor is massive
    def __init__(self, class_id_to_frequency: Dict[int, int], total_samples: int) -> None:
        self.total_samples = total_samples
        self.num_classes = len(class_id_to_frequency)
        self.class_id_to_frequency = class_id_to_frequency
        self.ignore_empty_annotations = True
        self._setup_data_source()
        super().__init__()

    @property
    def _all_classes(self):
        return [f"class_{i}" for i in range(self.num_classes)]

    def _setup_data_source(self) -> int:
        flattened_list = list()
        for k, v in self.class_id_to_frequency.items():
            flattened_list.extend([k] * v)

        random.shuffle(flattened_list)

        self.idx_to_classes = np.array_split(flattened_list, self.total_samples)
        return len(self.idx_to_classes)

    def _load_sample_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        targets = self.idx_to_classes[sample_id][..., np.newaxis]
        return {"image_path": "dummy.png", "target": targets}

    def __len__(self) -> int:
        return len(self.idx_to_classes)

    def __getitem__(self, idx: int):
        return torch.rand(1, 3, 16, 16), self._load_sample_annotation(idx)["target"]


class DatasetIndexMappingTest(unittest.TestCase):
    def test_balancing_classes_that_are_with_same_frequency(self):
        id_to_freq = {0: 10, 1: 10, 2: 10}
        n_samplers = 2
        n_samples = n_samplers * 7
        dataset = DummyDetectionDataset(class_id_to_frequency=id_to_freq, total_samples=n_samples)

        samplers = [
            DetectionClassBalancedDistributedSampler(
                dataset=dataset,  # noqa
                oversample_threshold=None,
                shuffle=False,
                num_replicas=2,
                rank=r,
                drop_last=False,  # requires len(dataset) % num_replicas == 0
            )
            for r in [0, 1]
        ]

        indices_sampled = []

        for sampler in samplers:
            for sample in sampler:
                indices_sampled.append(sample)

        self.assertSetEqual(set(indices_sampled), set(range(n_samples)))  # we sampled uniformly all the indices

    def test_balancing_scarce_classes(self):
        id_to_freq = {0: 10, 1: 3, 2: 10}  # class 1 is scarce, appears 3 / 20. Other classes are 10/20.
        n_samplers = 2
        n_samples = n_samplers * 10
        dataset = DummyDetectionDataset(class_id_to_frequency=id_to_freq, total_samples=n_samples)

        samplers = [
            DetectionClassBalancedDistributedSampler(
                dataset=dataset,
                oversample_threshold=None,
                shuffle=False,
                num_replicas=2,
                rank=r,
                drop_last=False,  # requires len(dataset) % num_replicas == 0
            )
            for r in [0, 1]
        ]

        indices_sampled = []

        for sampler in samplers:
            for sampled_idx in sampler:
                indices_sampled.append(sampled_idx)

        new_distribution = {0: 0, 1: 0, 2: 0}
        for i in indices_sampled:
            _, labels = samplers[0].dataset[i]  # can't use dataset[i] because we wrap it with oversampling mapping
            for label in labels:
                new_distribution[int(label)] += 1

        self.assertTrue(len(indices_sampled) > n_samples)
        self.assertTrue(new_distribution[1] / len(indices_sampled) > id_to_freq[1] / n_samples)


if __name__ == "__main__":
    unittest.main()
