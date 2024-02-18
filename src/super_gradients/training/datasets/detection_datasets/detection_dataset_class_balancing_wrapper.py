import math
from functools import partial
from typing import Optional, List
from torch.utils.data import DistributedSampler

from super_gradients.common.object_names import Samplers
from super_gradients.common.registry import register_sampler
from super_gradients.training.datasets import DetectionDataset
from super_gradients.training.datasets.balancing_classes_utils import get_repeat_factors, IndexMappingDatasetWrapper


def extract_labels_from_target(dataset: DetectionDataset, idx: int) -> List[int]:
    return dataset._load_sample_annotation(idx)["target"][:, -1]


@register_sampler(Samplers.DISTRIBUTED_DETECTION_CLASS_BALANCING)
class DetectionClassBalancedDistributedSampler(DistributedSampler):
    """
    DetectionClassBalancedDistributedSampler is a distributed sampler that over-samples scarce classes in detection datasets.
    """

    def __init__(self, dataset: DetectionDataset, oversample_threshold: Optional[float] = None, *args, **kwargs) -> None:
        repeat_factors = get_repeat_factors(
            index_to_classes=partial(extract_labels_from_target, dataset),
            num_classes=len(dataset._all_classes),
            dataset_length=len(dataset),
            ignore_empty_annotations=dataset.ignore_empty_annotations,
            oversample_threshold=oversample_threshold,
        )
        repeat_indices = []
        for dataset_idx, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_idx] * math.ceil(repeat_factor))

        super().__init__(dataset=IndexMappingDatasetWrapper(dataset, repeat_indices), *args, **kwargs)
