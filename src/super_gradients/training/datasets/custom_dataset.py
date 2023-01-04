import random
from typing import Callable, Dict, Tuple, Any, List, Sequence
import numpy as np

from torch.utils.data.dataset import Dataset
from super_gradients.training.transforms.transforms import Transform
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.datasets.datasets_conf import PASCAL_VOC_2012_CLASSES_LIST


class CustomDataset(Dataset):
    """Wrap any dataset to support SG transforms."""

    def __init__(self, dataset, transforms: List[Transform], input_adapters: Dict[str, Callable[[Any], np.ndarray]], fields: Sequence[str]):
        """
        :param dataset:
        :param transforms:
        :param input_adapters:  Dictionary of adapters, that takes every field from dataset.__getitem__ and converts them to np.ndarray
        :param fields:         fields
        """
        self.dataset = dataset
        self.transforms = transforms
        self.input_adapters = input_adapters
        self.fields = fields

    def __getitem__(self, item: Tuple[Any]) -> Tuple[np.ndarray]:
        """Wraps the original dataset and apply SG transform over it."""
        items = self.dataset[item]
        if len(items) != len(self.fields):
            raise RuntimeError(f"The dataset {self.dataset.__name__} is expected to return {len(self.fields)} items, but instead returns {len(items)}")

        sample = self._adapt_inputs(items)
        sample = self._apply_transforms(sample)

        return tuple(sample[field] for field in self.fields)

    def _adapt_inputs(self, items: Tuple[Any]) -> Dict[str, np.ndarray]:
        """Transform the dataset output items into a dictionary of np.ndarray.
        This is required in order to apply SG transforms on the dataset items."""
        sample = tuple_to_dict(self.fields, items)
        for field, adapter in self.input_adapters.items():
            if adapter:
                sample[field] = adapter(sample[field])
            if not isinstance(sample[field], np.ndarray):
                if adapter:
                    raise TypeError(f"The output of your {field} adapter is {type(sample[field])} when it should be np.ndarray.")
                else:
                    raise TypeError(
                        f"The output of the field {field} of your dataset is {type(sample[field])} when it should be np.ndarray."
                        f"Feel free to add an adapter function into input_adapters[{field}] so that it returns np.ndarray."
                    )
        return sample

    def _apply_transforms(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for transform in self.transforms:
            if hasattr(transform, "additional_samples_count"):
                # Get raw samples, but without any Transform
                new_raw_samples = get_random_samples(dataset=self.dataset, count=transform.additional_samples_count)
                additional_samples = [self._adapt_inputs(items) for items in new_raw_samples]
                sample = transform(sample={**sample, "additional_samples": additional_samples})
            else:
                sample = transform(sample=sample)
        return sample

    def __len__(self):
        return len(self.dataset)


def tuple_to_dict(fields: Sequence[str], items: Tuple[Any]) -> Dict[str, Any]:
    return {field: val for field, val in zip(fields, items)}


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
    return CustomDataset(dataset=dataset, transforms=transforms, input_adapters=adapters, fields=adapters.keys())


def parse_pascal_target(img_annotations: dict) -> np.ndarray:

    annotations = img_annotations["annotation"]["object"]
    target = np.zeros((len(annotations), 5))
    for ix, annotation in enumerate(annotations):
        bbox = annotation["bndbox"]
        target[ix, 0:4] = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])
        cls_id = PASCAL_VOC_2012_CLASSES_LIST.index(annotation["name"])
        target[ix, 4] = cls_id
    return target
