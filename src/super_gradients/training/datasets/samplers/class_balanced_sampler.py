import json
import os
import warnings
from typing import List, Optional

import numpy as np
from torch.utils.data import WeightedRandomSampler

from super_gradients.common.object_names import Samplers
from super_gradients.common.registry import register_sampler
from super_gradients.dataset_interfaces import HasClassesInformation


def _default_oversample_heuristic(
    class_frequencies: np.ndarray, oversample_threshold: Optional[float] = None, oversample_aggressiveness: float = 0.5
) -> np.ndarray:
    """
    How to oversample a class. This heuristic is based on https://arxiv.org/pdf/1908.03195.pdf.
    :param class_frequencies:           A numpy array of class frequencies.
    :param oversample_threshold:        A frequency threshold (fraction, 0-1). Classes that are *less frequent* than this threshold will be oversampled.
                                        The default value is None. If None, the median of the class frequencies will be used.
    :param oversample_aggressiveness:   How aggressive the oversampling is. The higher the value, the more aggressive the oversampling is.
                                        The default value is 0.5, and corresponds to the implementation in the paper.
                                        A value of 0.0 corresponds to no oversampling.

    Returns a numpy array indicating the oversample factor per class. An entry with value of 1.0 corresponds to no oversampling.
    """
    if oversample_threshold is None:
        oversample_threshold = np.median(class_frequencies)

    result = np.ones_like(class_frequencies, dtype=np.float32)
    for cls, cls_freq in enumerate(class_frequencies):
        if cls_freq == 0 or cls_freq > oversample_threshold:
            continue
        result[cls] = (oversample_threshold / cls_freq) ** oversample_aggressiveness

    return result


class ClassBalancer:
    @staticmethod
    def get_sample_repeat_factors(
        class_information_provider: HasClassesInformation,
        oversample_threshold: Optional[float] = None,
        oversample_aggressiveness: float = 0.5,
    ) -> List[float]:
        """
        Oversampling scarce classes from detection dataset, following sampling strategy described in https://arxiv.org/pdf/1908.03195.pdf.

        :param class_information_provider:      An object (probably a dataset) that provides the class information.
        :param oversample_threshold:            A frequency threshold (fraction, 0-1). Classes that are *less frequent* than this threshold will be oversampled.
                                                The default value is None. If None, the median of the class frequencies will be used.
        :param oversample_aggressiveness:       How aggressive the oversampling is. The higher the value, the more aggressive the oversampling is.
                                                The default value is 0.5, and corresponds to the implementation in the paper.
                                                A value of 0.0 corresponds to no oversampling.


        The repeat factor is computed as followed:
        1. For each class c, compute the fraction # of images that contain it (its frequency): :math:`f(c)`
        2. For each class c, compute the category-level repeat factor: :math:`r(c) = max(1, aggressiveness(threshold/f(c)))`
        3. For each image I, compute the image-level repeat factor: :math:`r(I) = max_{c in I} r(c)`

        Returns a list of repeat factors (length = dataset_length). How to read: result[i] is a float, indicates the repeat factor of image i.
        """

        class_information = class_information_provider.get_dataset_classes_information()  # shape = (dataset_length, num_classes)

        # 1. For each category c, compute the fraction # of images that contain it: f(c)
        class_frequencies = np.sum(class_information, axis=0)
        class_frequencies = class_frequencies / len(class_information)

        # 2. For each class c, compute the class-level repeat factor: r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: cat_repeat
            for cat_id, cat_repeat in enumerate(_default_oversample_heuristic(class_frequencies, oversample_threshold, oversample_aggressiveness))
        }  # dict for ease of debugging

        # 3. For each image I, compute the image-level repeat factor: r(I) = max_{c in I} r(c)
        repeat_factors = list()
        categories = np.arange(class_information.shape[1])
        for sample_cat_freq in class_information:
            cat_ids = categories[sample_cat_freq != 0]
            if len(cat_ids) == 0:  # in case image doesn't have annotations, we will not over-sample nor ignore it
                repeat_factors.append(1.0)
            else:
                repeat_factors.append(max({category_repeat[cat_id] for cat_id in cat_ids}))

        return repeat_factors  # len = dataset_length

    @staticmethod
    def precompute_sample_repeat_factors(
        output_path: str,
        class_information_provider: HasClassesInformation,
        oversample_threshold: Optional[float] = None,
    ):
        repeat_factors: List[float] = ClassBalancer.get_sample_repeat_factors(
            class_information_provider=class_information_provider,
            oversample_threshold=oversample_threshold,
        )

        str_repeat_factors = [np.format_float_positional(rf, trim="0", precision=4) for rf in repeat_factors]

        with open(output_path, "w") as f:
            json.dump(str_repeat_factors, f)

    @staticmethod
    def from_precomputed_sample_repeat_factors(precomputed_path: str) -> List[float]:
        """
        Loads the repeat factors from a precomputed file.
        """
        if not os.path.exists(precomputed_path):
            raise FileNotFoundError(f"`{precomputed_path}` does not exist.")

        with open(precomputed_path, "r") as f:
            loaded = json.load(f)

        return list(map(lambda x: float(x), loaded))


@register_sampler(Samplers.CLASS_BALANCED)
class ClassBalancedSampler(WeightedRandomSampler):
    def __init__(
        self,
        dataset: Optional[HasClassesInformation] = None,
        precomputed_factors_file: Optional[str] = None,
        oversample_threshold: Optional[float] = None,
        oversample_aggressiveness: float = 0.5,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        """
        Wrap WeightedRandomSampler with weights that are computed from the class frequencies of the dataset.
        """

        if dataset is None and precomputed_factors_file is None:
            raise ValueError("`dataset` and `precomputed_factors` cannot be both None.")

        if dataset is not None and precomputed_factors_file is not None:
            # this logic is to simplify `_instantiate_sampler` method.
            warnings.warn("Both `dataset` and `precomputed_factors_file` are passed. `dataset` WILL BE IGNORED!")

        if precomputed_factors_file is not None:
            repeat_factors = ClassBalancer.from_precomputed_sample_repeat_factors(precomputed_factors_file)
        else:
            if not isinstance(dataset, HasClassesInformation):
                raise ValueError(f"`dataset` must be an instance of `{HasClassesInformation.__name__}`.")

            repeat_factors = ClassBalancer.get_sample_repeat_factors(dataset, oversample_threshold, oversample_aggressiveness)

        weights = np.array(repeat_factors) / sum(repeat_factors)

        super().__init__(weights=weights, num_samples=num_samples or len(weights), replacement=True, generator=generator)
