import random
from typing import Callable, Dict, Tuple, Any
import numpy as np

from torch.utils.data.dataset import Dataset
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.datasets.datasets_conf import PASCAL_VOC_2012_CLASSES_LIST


class CustomDataset(Dataset):
    """Wrap any dataset to be support SG transforms."""

    def __init__(self, dataset, transforms, input_adapters: Dict[str, Callable], columns: Tuple[str]):
        """

        :param dataset:
        :param transforms:
        :param input_adapters: Dictionary of adapters, that takes every column from dataset.__getitem__ and converts them to np.ndarray
        :param columns:
        """
        self.dataset = dataset
        self.transforms = transforms
        self.input_adapters = input_adapters
        self.columns = columns

    def __getitem__(self, item):
        items = self.dataset[item]
        if len(items) != len(self.columns):
            raise RuntimeError(f"The dataset {self.dataset.__name__} is expected to return {len(self.columns)} items, but instead returns {len(items)}")

        sample = self._adapt_inputs(items)

        for transform in self.transforms:
            if hasattr(transform, "additional_samples_count"):
                # Get raw samples, but without any Transform
                new_raw_samples = get_random_samples(dataset=self.dataset, count=transform.additional_samples_count)
                additional_samples = [self._adapt_inputs(items) for items in new_raw_samples]
                sample = transform(sample={**sample, "additional_samples": additional_samples})
            else:
                sample = transform(sample=sample)
        return tuple(sample[col] for col in self.columns)

    def _tuple_to_dict(self, items: Tuple[Any]) -> Dict[str, Any]:
        return {col: val for col, val in zip(self.columns, items)}

    def _adapt_inputs(self, items: Tuple[Any]) -> Dict[str, np.ndarray]:
        sample = self._tuple_to_dict(items)
        for col, adapter in self.input_adapters.items():
            if adapter:
                sample[col] = adapter(sample[col])
            if not isinstance(sample[col], np.ndarray):
                if adapter:
                    raise TypeError(f"The output of your {col} adapter is {type(sample[col])} when it should be np.ndarray.")
                else:
                    raise TypeError(
                        f"The output of the column {col} of your dataset is {type(sample[col])} when it should be np.ndarray."
                        f"Feel free to add an adapter function into input_adapters[{col}] so that it returns np.ndarray."
                    )
        return sample

    def __len__(self):
        return len(self.dataset)


def get_random_sample(dataset):
    return dataset[random.randint(0, len(dataset) - 1)]


def get_random_samples(dataset, count: int) -> list:
    """Load random samples.

    :param count: The number of samples wanted
    :param non_empty_annotations_only: If true, only return samples with at least 1 annotation
    :return: A list of samples satisfying input params
    """
    return [get_random_sample(dataset) for _ in range(count)]


@resolve_param("transforms", factory=TransformsFactory())
def wrap_detection_dataset(dataset, transforms, image_adapter=None, target_adapter=None):
    """
    WARNING: dataset.__getitem__ should return a tuple (img, target)!
    :param dataset:         Dataset to wrap.
    :param transforms:      SuperGradients transforms
    :param image_adapter:   Optional. Adapter that takes dataset.__getitem__ and converts it to np.ndarray
    :param target_adapter:  Optional. Adapter that takes dataset.__getitem__ and converts it to np.ndarray
    :return:
    """
    adapters = {"image": image_adapter, "target": target_adapter}
    return CustomDataset(dataset=dataset, transforms=transforms, input_adapters=adapters, columns=adapters.keys())


def parse_pascal_target(img_annotations: dict) -> np.ndarray:

    annotations = img_annotations["annotation"]["object"]
    target = np.zeros((len(annotations), 5))
    for ix, annotation in enumerate(annotations):
        bbox = annotation["bndbox"]
        target[ix, 0:4] = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])
        cls_id = PASCAL_VOC_2012_CLASSES_LIST.index(annotation["name"])
        target[ix, 4] = cls_id
    return target
