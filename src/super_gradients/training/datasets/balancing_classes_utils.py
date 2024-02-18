import math
from typing import Callable, List, Optional

import numpy as np
from torch.utils.data import Dataset


def _default_oversample_heuristic(cat_id, cat_freq, oversample_threshold):
    """
    How to oversample a category. This heuristic is based on https://arxiv.org/pdf/1908.03195.pdf.
    A value of 1.0 corresponds to no oversampling.
    """
    return max(1.0, math.sqrt(oversample_threshold / cat_freq))


def get_repeat_factors(
    *,
    index_to_classes: Callable[[int], List[int]],
    num_classes: int,
    dataset_length: int,
    ignore_empty_annotations: bool = False,
    oversample_threshold: Optional[float] = None,
    oversample_heuristic: Optional[Callable[[int, int, float], float]] = _default_oversample_heuristic,
) -> List[float]:
    """
    Oversampling scarce classes from detection dataset, following sampling strategy described in https://arxiv.org/pdf/1908.03195.pdf.
    Implementation based on https://github.com/open-mmlab/mmdetection (Apache 2.0 license).

    :param index_to_classes: A function that maps an index to a list of classes (labels).
    :param num_classes: Number of classes.
    :param dataset_length: Length of the dataset.
    :param ignore_empty_annotations: Whether to ignore empty annotations.
    :param oversample_threshold: A frequency threshold (fraction, 0-1). Classes that are *less frequent* than this threshold will be oversampled.
                                    Default: None. If None, the median of the class frequencies will be used.
                                    See step (1) below to understand how frequencies are computed.


    The repeat factor is computed as followed.
    1. For each category c, compute the fraction # of images that contain it (its frequency): :math:`f(c)`
    2. For each category c, compute the category-level repeat factor: :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor: :math:`r(I) = max_{c in I} r(c)`

    Returns a list of repeat factors (length = dataset_length). How to read: result[i] is a float, indicates the repeat factor of image i.
    """

    # 0. pre-fetch the categories to reduce time
    sample_id_to_categories = {idx: index_to_classes(idx) for idx in range(dataset_length)}

    # 1. For each category c, compute the fraction # of images that contain it: f(c)
    category_freq = dict()
    for idx in range(dataset_length):
        cat_ids = sample_id_to_categories[idx]
        if len(cat_ids) == 0 and not ignore_empty_annotations:
            cat_ids = {num_classes}
        for cat_id in cat_ids:
            category_freq[cat_id] = category_freq.get(cat_id, 0) + 1
    for k, v in category_freq.items():
        category_freq[k] = v / dataset_length

    # 2. For each category c, compute the category-level repeat factor: r(c) = max(1, sqrt(t/f(c)))
    if oversample_threshold is None:
        oversample_threshold = np.median(list(category_freq.values()))
    category_repeat = {cat_id: oversample_heuristic(cat_id, cat_freq, oversample_threshold) for cat_id, cat_freq in category_freq.items()}

    # 3. For each image I, compute the image-level repeat factor: r(I) = max_{c in I} r(c)
    repeat_factors = []
    for idx in range(dataset_length):
        cat_ids = sample_id_to_categories[idx]
        if len(cat_ids) == 0:  # no annotations for this image
            repeat_factors.append(1.0 * (not ignore_empty_annotations))  # 0.0 if we ignore, 1.0 otherwise. We are not oversampling empty annotations.
            continue
        repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
        repeat_factors.append(repeat_factor)

    return repeat_factors


class IndexMappingDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, mapping: List[int]):
        super().__init__()
        self.dataset = dataset
        self.mapping = mapping

    def __getitem__(self, item):
        return self.dataset[self.mapping[item]]

    def __len__(self):
        return len(self.mapping)
