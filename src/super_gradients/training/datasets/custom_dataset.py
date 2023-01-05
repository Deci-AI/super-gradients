import random
from typing import Callable, Dict, Tuple, Any, List, Sequence
import numpy as np

from torch.utils.data.dataset import Dataset
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.transforms.transforms import Transform
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.datasets_factory import DatasetsFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.datasets.datasets_conf import PASCAL_VOC_2012_CLASSES_LIST


import torch.utils.data.dataset
import torchvision.transforms as T


class ApplyPerField(Transform):
    def __init__(self, transforms: dict):
        self.transforms = transforms

    def __call__(self, sample):
        for col, col_transform in self.transforms.items():
            sample[col] = col_transform(sample[col])
        return sample


class CustomDataset(Dataset):
    """Wrap any dataset to support SG transforms."""

    @resolve_param("transforms", factory=TransformsFactory())
    @resolve_param("dataset", factory=DatasetsFactory())
    def __init__(self, dataset, transforms: List[Transform], fields: Sequence[str], input_adapters: Dict[str, Callable[[Any], np.ndarray]] = None):
        """
        :param dataset:         Dataset to wrap.
        :param transforms:      Transforms
        :param input_adapters:  Dictionary of adapters, that takes every field from dataset.__getitem__ and converts them to np.ndarray
        :param fields:          Name of the output fields of the dataset.
                                    - The order should match the output of the wrapped dataset
                                    - The names should match the fields used in the transform ("image", "target", "mask", ...)
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
        for field in self.fields:
            adapter = self.input_adapters.get(field)
            if adapter:
                sample[field] = adapter(sample[field])
            # if not isinstance(sample[field], np.ndarray):
            #     if adapter:
            #         raise TypeError(f"The output of your {field} adapter is {type(sample[field])} when it should be np.ndarray.")
            #     else:
            #         raise TypeError(
            #             f"The output of the field {field} of your dataset is {type(sample[field])} when it should be np.ndarray."
            #             f"Feel free to add an adapter function into input_adapters[{field}] so that it returns np.ndarray."
            #         )
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


class CustomDetectionDataset(CustomDataset):
    def __init__(self, dataset, transforms: List[Transform], image_adapter=None, target_adapter=None):
        super().__init__(
            dataset=dataset,
            transforms=transforms,
            field=("image", "target"),
            input_adapters={"image": image_adapter, "target": target_adapter},
        )


class CustomSegmentationDataset(CustomDataset):
    def __init__(self, dataset, transforms: List[Transform], image_adapter=None, mask_adapter=None):

        # This is because currently the Segmentation classes apply this behind the hood. So to mimic the behavior we also do it here.
        # TODO: This should be moved to transforms.s
        def post_processing_mask(mask):
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask == 255] = 19
            return mask

        post_processing = ApplyPerField(
            {"image": T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), "mask": post_processing_mask}
        )

        super().__init__(
            dataset=dataset,
            transforms=transforms + [post_processing],
            fields=("image", "mask"),
            input_adapters={"image": image_adapter, "mask": mask_adapter},
        )


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


def register_detection_dataset(dataset, register_as: str, image_adapter=None, target_adapter=None):
    """
    WARNING: dataset.__getitem__ should return a tuple (img, target)!
    :param image_adapter:   Optional. Adapter that takes dataset.__getitem__ and converts it to np.ndarray
    :param target_adapter:  Optional. Adapter that takes dataset.__getitem__ and converts it to np.ndarray
    :return:
    """

    @register_dataset(register_as)
    def custom_dataset(transforms):
        return CustomDataset(
            dataset=dataset,
            transforms=transforms,
            input_adapters={"image": image_adapter, "target": target_adapter},
            fields=("image", "target"),
        )

    return


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
    return CustomDataset(dataset=dataset, transforms=transforms, input_adapters=adapters, fields=list(adapters.keys()))


@resolve_param("transforms", factory=TransformsFactory())
def wrap_segmentation_dataset(dataset, transforms):
    import numpy as np

    def process_target(target):
        target = torch.from_numpy(np.array(target)).long()
        target[target == 255] = 19
        return target

    class TransformDict:
        def __init__(self, transforms: dict):
            self.transforms = transforms

        def __call__(self, sample):
            for col, col_transform in self.transforms.items():
                sample[col] = col_transform(sample[col])
            return sample

    custom_transform = TransformDict({"image": T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), "mask": process_target})

    transforms += [custom_transform]
    return CustomDataset(dataset=dataset, transforms=transforms, input_adapters={}, fields=("image", "mask"))


def parse_pascal_target(img_annotations: dict) -> np.ndarray:

    annotations = img_annotations["annotation"]["object"]
    target = np.zeros((len(annotations), 5))
    for ix, annotation in enumerate(annotations):
        bbox = annotation["bndbox"]
        target[ix, 0:4] = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])
        cls_id = PASCAL_VOC_2012_CLASSES_LIST.index(annotation["name"])
        target[ix, 4] = cls_id
    return target
