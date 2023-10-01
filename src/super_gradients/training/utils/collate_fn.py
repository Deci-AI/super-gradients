from torch.utils.data.dataloader import default_collate
import random
from typing import List, Union, Tuple, Optional, Dict, Type, Callable, Iterable, Sequence

from abc import ABC, abstractmethod

import cv2
import torch
import numpy as np


from data_gradients.dataset_adapters.config.data_config import DataConfig, DetectionDataConfig, SegmentationDataConfig, ClassificationDataConfig
from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType

from super_gradients.common.registry.registry import register_collate_function
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.collate_functions_factory import CollateFunctionsFactory


class DatasetItemsException(Exception):
    def __init__(self, data_sample: Tuple, collate_type: Type, expected_item_names: Tuple):
        """
        :param data_sample: item(s) returned by a data
        :param collate_type: type of the collate that caused the exception
        :param expected_item_names: tuple of names of items that are expected by the collate to be returned from the data
        """
        collate_type_name = collate_type.__name__
        num_sample_items = len(data_sample) if isinstance(data_sample, tuple) else 1
        error_msg = f"`{collate_type_name}` only supports Datasets that return a tuple {expected_item_names}, but got a tuple of len={num_sample_items}"
        super().__init__(error_msg)


class BaseDatasetAdapterCollateFN(ABC):
    """Base Collate function that adapts an input data to SuperGradients format

    This is done by applying the adapter logic either before or after the original collate function,
    depending on whether the adapter was set up on a batch or a sample.

    Note that the original collate function (if any) will still be used, but will be wrapped into this class.
    """

    @resolve_param("base_collate_fn", CollateFunctionsFactory())
    def __init__(self, adapter: BaseDatasetAdapter, base_collate_fn: Callable):
        """
        :param base_collate_fn:     Collate function to wrap. If None, the default collate function will be used.
        :param adapter:             Dataset adapter to use
        """
        self._adapt_on_batch = adapter.data_config.is_batch

        self.adapter = adapter
        self._base_collate_fn = base_collate_fn

        if isinstance(self._base_collate_fn, type(self)):
            raise RuntimeError(f"You just tried to instantiate {self.__class__.__name__} with a `base_collate_fn` of the same type, which is not supported.")

    def __call__(self, samples: Iterable[SupportedDataType]) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._require_setup:
            # This is required because python `input` is no compatible multiprocessing (e.g. `num_workers > 0`, or `DDP`)
            # And if not `self._require_setup`, the adapter will need to ask at least one question using the python `input`
            raise RuntimeError(
                f"Trying to collate using `{self.__class__.__name__}`, but it was not fully set up yet. Please do one of the following\n"
                f"   - Call `{self.__class__.__name__}(...).setup_adapter(dataloader)` before iterating over the dataloader.\n"
                f"   - or Instantiate `{self.__class__.__name__}(adapter_cache_path=...)` with `adapter_cache_path` mapping to the cache file of "
                f"an adapter that was already set up on this data.\n"
            )

        if not self._adapt_on_batch:
            samples = self._adapt_samples(samples=samples)

        batch = self._base_collate_fn(samples)

        if self._adapt_on_batch:
            batch = self._adapt_batch(batch=batch)

        images, targets = batch  # At this point we know it is (images, targets) because the adapter was used - either on samples or batch
        return images, targets

    @property
    def _require_setup(self) -> bool:
        return not self.adapter.data_config.is_completely_initialized

    def _adapt_samples(self, samples: Iterable[SupportedDataType]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the adapter logic to a list of samples. This should be called only if the adapter was NOT setup on a batch.
        :param samples: List of samples to adapt
        :return:        List of (Image, Targets)
        """
        adapted_samples = []
        for sample in samples:
            images, targets = self.adapter.adapt(sample)  # Will construct batch of 1
            images, targets = images.squeeze(0), targets.squeeze(0)  # Extract the sample
            adapted_samples.append((images, targets))
        return adapted_samples

    def _adapt_batch(self, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the adapter logic to a batch. This should be called only if the adapter was setup on a batch.
        :param batch: Batch of samples to adapt
        :return:      Adapted batch (Images, Targets)
        """
        images, targets = self.adapter.adapt(batch)
        return images, targets


@register_collate_function()
class DetectionDatasetAdapterCollateFN(BaseDatasetAdapterCollateFN):
    """Detection Collate function that adapts an input data to SuperGradients format

    This is done by applying the adapter logic either before or after the original collate function,
    depending on whether the adapter was set up on a batch or a sample.

    Note that the original collate function (if any) will still be used, but will be wrapped into this class.
    """

    @resolve_param("base_collate_fn", CollateFunctionsFactory())
    def __init__(
        self, adapter_config: Optional[DetectionDataConfig] = None, adapter_cache_path: Optional[str] = None, base_collate_fn: Optional[Callable] = None
    ):
        """
        :param base_collate_fn:     Collate function to wrap. If None, the default collate function will be used.
        :param adapter_config:      Dataset adapter to use. Mutually exclusive with `adapter_cache_path`.
        :param adapter_cache_path:  Path to the cache file. Mutually exclusive with `adapter`.
        """
        if adapter_config and adapter_cache_path:
            raise ValueError("`adapter_config` and `adapter_cache_path` cannot be set at the same time.")
        elif adapter_config is None and adapter_cache_path:
            adapter = DetectionDatasetAdapter.from_cache(cache_path=adapter_cache_path)
        elif adapter_config is not None and adapter_cache_path is None:
            adapter = DetectionDatasetAdapter(data_config=adapter_config)
        else:
            raise ValueError("Please either set `adapter_config` or `adapter_cache_path`.")

        # `DetectionCollateFN()` is the default collate_fn for detection.
        # But if the adapter was used on already collated batches, we don't want to force it.
        base_collate_fn = base_collate_fn or (default_collate if adapter.data_config.is_batch else DetectionCollateFN())
        super().__init__(adapter=adapter, base_collate_fn=base_collate_fn)

    def _adapt_samples(self, samples: Iterable[SupportedDataType]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Apply the adapter logic to a list of samples. This should be called only if the adapter was NOT setup on a batch.
        :param samples: List of samples to adapt
        :return:        List of (Image, Targets)
        """
        from super_gradients.training.utils.detection_utils import xyxy2cxcywh

        adapted_samples = []
        for sample in samples:
            images, targets = self.adapter.adapt(sample)  # Will construct batch of 1
            images, targets = images[0], targets[0]  # Extract the sample
            targets[:, 1:] = xyxy2cxcywh(targets[:, 1:])
            adapted_samples.append((images, targets))
        return adapted_samples

    def _adapt_batch(self, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        from super_gradients.training.utils.detection_utils import xyxy2cxcywh

        images, targets = super()._adapt_batch(batch)
        targets = DetectionCollateFN._format_targets(targets)
        # This is only relevant if we adapt on the batch.
        # targets = ensure_flat_bbox_batch(targets)  # If adapter applied on collated batch, the output will be (BS, P, 5). In such case, we want -> (N, 6)
        targets[:, 2:] = xyxy2cxcywh(targets[:, 2:])  # Adapter returns xyxy
        return images, targets


@register_collate_function()
class SegmentationDatasetAdapterCollateFN(BaseDatasetAdapterCollateFN):
    """Segmentation Collate function that adapts an input data to SuperGradients format

    This is done by applying the adapter logic either before or after the original collate function,
    depending on whether the adapter was set up on a batch or a sample.

    Note that the original collate function (if any) will still be used, but will be wrapped into this class.
    """

    @resolve_param("base_collate_fn", CollateFunctionsFactory())
    def __init__(
        self, adapter_config: Optional[SegmentationDataConfig] = None, adapter_cache_path: Optional[str] = None, base_collate_fn: Optional[Callable] = None
    ):
        """
        :param base_collate_fn:     Collate function to wrap. If None, the default collate function will be used.
        :param adapter_config:      Dataset adapter to use. Mutually exclusive with `adapter_cache_path`.
        :param adapter_cache_path:  Path to the cache file. Mutually exclusive with `adapter`.
        """
        if adapter_config and adapter_cache_path:
            raise ValueError("`adapter_config` and `adapter_cache_path` cannot be set at the same time.")
        elif adapter_config is None and adapter_cache_path:
            adapter = SegmentationDatasetAdapter.from_cache(cache_path=adapter_cache_path)
        elif adapter_config is not None and adapter_cache_path is None:
            adapter = SegmentationDatasetAdapter(data_config=adapter_config)
        else:
            raise ValueError("Please either set `adapter_config` or `adapter_cache_path`.")

        super().__init__(adapter=adapter or base_collate_fn, base_collate_fn=base_collate_fn)

    def __call__(self, samples: Iterable[SupportedDataType]) -> Tuple[torch.Tensor, torch.Tensor]:
        from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet

        images, targets = super().__call__(samples=samples)  # This already returns a batch of (images, targets)
        transform = SegmentationDataSet.get_normalize_transform()
        images = transform(images / 255)  # images are [0-255] after the data adapter
        targets = targets.argmax(1)
        return images, targets


@register_collate_function()
class ClassificationDatasetAdapterCollateFN(BaseDatasetAdapterCollateFN):
    """Classification Collate function that adapts an input data to SuperGradients format

    This is done by applying the adapter logic either before or after the original collate function,
    depending on whether the adapter was set up on a batch or a sample.

    Note that the original collate function (if any) will still be used, but will be wrapped into this class.
    """

    @resolve_param("base_collate_fn", CollateFunctionsFactory())
    def __init__(
        self, adapter_config: Optional[ClassificationDataConfig] = None, adapter_cache_path: Optional[str] = None, base_collate_fn: Optional[Callable] = None
    ):
        """
        :param base_collate_fn:     Collate function to wrap. If None, the default collate function will be used.
        :param adapter_config:      Dataset adapter to use. Mutually exclusive with `adapter_cache_path`.
        :param adapter_cache_path:  Path to the cache file. Mutually exclusive with `adapter`.
        """
        if adapter_config and adapter_cache_path:
            raise ValueError("`adapter_config` and `adapter_cache_path` cannot be set at the same time.")
        elif adapter_config is None and adapter_cache_path:
            adapter = ClassificationDatasetAdapter.from_cache(cache_path=adapter_cache_path)
        elif adapter_config is not None and adapter_cache_path is None:
            adapter = ClassificationDatasetAdapter(data_config=adapter_config)
        else:
            raise ValueError("Please either set `adapter_config` or `adapter_cache_path`.")

        super().__init__(adapter=adapter or base_collate_fn, base_collate_fn=base_collate_fn)

    def __call__(self, samples: Iterable[SupportedDataType]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = super().__call__(samples=samples)  # This already returns a batch of (images, targets)
        images = images / 255
        return images, targets


def ensure_flat_bbox_batch(bbox_batch: torch.Tensor) -> torch.Tensor:
    """
    Flatten a batched bounding box tensor and prepend the batch ID to each bounding box. Excludes padding boxes.

    :param bbox_batch: Bounding box tensor of shape (BS, PaddingSize, 5).
    :return: Flattened tensor of shape (N, 6), where N <= BS * PaddingSize.
    """

    # Create a tensor of batch IDs
    batch_ids = torch.arange(bbox_batch.size(0), device=bbox_batch.device).unsqueeze(-1)
    batch_ids = batch_ids.repeat(1, bbox_batch.size(1)).reshape(-1, 1)  # Shape: (BS*PaddingSize, 1)

    # Reshape bounding box tensor
    bbox_reshaped = bbox_batch.reshape(-1, 5)  # Shape: (BS*PaddingSize, 5)

    # Concatenate batch IDs and reshaped bounding boxes
    flat_bbox = torch.cat((batch_ids, bbox_reshaped), dim=1)  # Shape: (BS*PaddingSize, 6)

    # Filter out padding boxes (assuming padding boxes have all values zero)
    non_padding_mask = torch.any(flat_bbox[:, 1:] != 0, dim=1)
    flat_bbox = flat_bbox[non_padding_mask]

    return flat_bbox


class BaseDataloaderAdapter(ABC):
    @classmethod
    def from_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        adapter_config: Optional[DataConfig] = None,
        config_path: Optional[str] = None,
        collate_fn: Optional[callable] = None,
        **dataloader_kwargs,
    ) -> torch.utils.data.DataLoader:
        dataloader = torch.utils.data.DataLoader(dataset=dataset, **dataloader_kwargs)

        # `AdapterCollateFNClass` depends on the tasks, but just represents the collate function adapter for that specific task.
        AdapterCollateFNClass = cls._get_collate_fn_class()
        adapter_collate = AdapterCollateFNClass(base_collate_fn=collate_fn or default_collate, adapter_config=adapter_config, adapter_cache_path=config_path)

        _maybe_setup_adapter(adapter=adapter_collate.adapter, data=dataset)
        dataloader.collate_fn = adapter_collate
        return dataloader

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        adapter_config: Optional[DataConfig] = None,
        config_path: Optional[str] = None,
    ) -> torch.utils.data.DataLoader:
        # `AdapterCollateFNClass` depends on the tasks, but just represents the collate function adapter for that specific task.
        AdapterCollateFNClass = cls._get_collate_fn_class()
        adapter_collate = AdapterCollateFNClass(base_collate_fn=dataloader.collate_fn, adapter_config=adapter_config, adapter_cache_path=config_path)

        _maybe_setup_adapter(adapter=adapter_collate.adapter, data=dataloader)
        dataloader.collate_fn = adapter_collate
        return dataloader

    @classmethod
    @abstractmethod
    def _get_collate_fn_class(cls) -> type:
        """
        Returns the specific Collate Function class for this type of task.

        :return: Collate Function class specific to the task.
        """
        pass


def _maybe_setup_adapter(adapter: BaseDatasetAdapter, data: Iterable[SupportedDataType]) -> None:
    """Run a dummy iteration of a dataloader to make sure that the dataloader is properly adapted to SuperGradients format."""
    if not adapter.data_config.is_completely_initialized:
        from super_gradients.common.environment.ddp_utils import is_distributed

        if is_distributed():
            raise RuntimeError(
                f"`{adapter.__class__.__name__}` can be used with DDP only if the it was initialized BEFORE.\n"
                "   - If you already have a cache file from a previous run, please use it.\n"
                "   - Otherwise, please run your script WITHOUT DDP first, and then re-run this script on DDP using the cache file name."
            )  # TODO: Improve - make it more clear and explicit.
        _ = adapter.adapt(next(iter(data)))  # Run a dummy iteration to ensure all the questions are asked.
        adapter.data_config.dump_cache_file()


def maybe_setup_adapter_collate(dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    """If the dataloader collate function is an adapter, and requires to be set up, do it. Otherwise skip."""
    collate_fn = dataloader.collate_fn
    if isinstance(collate_fn, BaseDatasetAdapterCollateFN):
        if collate_fn.adapter.data_config.is_batch:
            # Enforce a first execution with 0 worker. This is required because python `input` is no compatible multiprocessing (i.e. num_workers > 0)
            # Therefore we want to make sure to ask the questions on 0 workers.
            dataloader.num_workers, _num_workers = 0, dataloader.num_workers
            _maybe_setup_adapter(adapter=collate_fn.adapter, data=dataloader)
            dataloader.num_workers = _num_workers
        else:
            _maybe_setup_adapter(adapter=collate_fn.adapter, data=dataloader.dataset)
    return dataloader


class DetectionDataloaderAdapter(BaseDataloaderAdapter):

    # TODO: Add `target_model` and adapt depending on the model.

    @classmethod
    def from_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        adapter_config: Optional[DataConfig] = None,
        config_path: Optional[str] = None,
        **dataloader_kwargs,
    ) -> torch.utils.data.DataLoader:
        return super().from_dataset(
            dataset=dataset, adapter_config=adapter_config, config_path=config_path, collate_fn=DetectionCollateFN(), **dataloader_kwargs
        )

    @classmethod
    def _get_collate_fn_class(cls) -> type:
        return DetectionDatasetAdapterCollateFN


class SegmentationDataloaderAdapter(BaseDataloaderAdapter):
    @classmethod
    def _get_collate_fn_class(cls) -> type:
        return SegmentationDatasetAdapterCollateFN


class ClassificationDataloaderAdapter(BaseDataloaderAdapter):
    @classmethod
    def _get_collate_fn_class(cls) -> type:
        return ClassificationDatasetAdapterCollateFN


@register_collate_function()
class DetectionCollateFN:
    """
    Collate function for Yolox training
    """

    def __init__(self):
        self.expected_item_names = ("image", "targets")

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            images_batch, labels_batch = list(zip(*data))
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=data[0], collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch)

    @staticmethod
    def _format_images(images_batch: List[Union[torch.Tensor, np.array]]) -> torch.Tensor:
        images_batch = [torch.tensor(img) for img in images_batch]
        images_batch_stack = torch.stack(images_batch, 0)
        if images_batch_stack.shape[3] == 3:
            images_batch_stack = torch.moveaxis(images_batch_stack, -1, 1).float()
        return images_batch_stack

    @staticmethod
    def _format_targets(labels_batch: List[Union[torch.Tensor, np.array]]) -> torch.Tensor:
        """
        Stack a batch id column to targets and concatenate
        :param labels_batch: a list of targets per image (each of arbitrary length)
        :return: one tensor of targets of all imahes of shape [N, 6], where N is the total number of targets in a batch
                 and the 1st column is batch item index
        """
        labels_batch = [torch.tensor(labels) for labels in labels_batch]
        labels_batch_indexed = []
        for i, labels in enumerate(labels_batch):
            batch_column = labels.new_ones((labels.shape[0], 1)) * i
            labels = torch.cat((batch_column, labels), dim=-1)
            labels_batch_indexed.append(labels)
        return torch.cat(labels_batch_indexed, 0)


@register_collate_function()
class CrowdDetectionCollateFN(DetectionCollateFN):
    """
    Collate function for Yolox training with additional_batch_items that includes crowd targets
    """

    def __init__(self):
        self.expected_item_names = ("image", "targets", "crowd_targets")
        super().__init__()

    def _format_batch(self, batch_data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            images_batch, labels_batch, crowd_labels_batch = batch_data
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=batch_data, collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch), {"crowd_targets": self._format_targets(crowd_labels_batch)}


@register_collate_function()
class PPYoloECollateFN(DetectionCollateFN):
    """
    Collate function for PPYoloE training
    """

    def __init__(self, random_resize_sizes: Union[List[int], None] = None, random_resize_modes: Union[List[int], None] = None):
        """
        :param random_resize_sizes: (rows, cols)
        """
        super().__init__()
        self.random_resize_sizes = random_resize_sizes
        self.random_resize_modes = random_resize_modes

    def __repr__(self):
        return f"PPYoloECollateFN(random_resize_sizes={self.random_resize_sizes}, random_resize_modes={self.random_resize_modes})"

    def __str__(self):
        return self.__repr__()

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random_resize_sizes is not None:
            data = self.random_resize(data)
        return super().__call__(data)

    def random_resize(self, batch):
        target_size = random.choice(self.random_resize_sizes)
        interpolation = random.choice(self.random_resize_modes)
        batch = [self.random_resize_sample(sample, target_size, interpolation) for sample in batch]
        return batch

    def random_resize_sample(self, sample, target_size, interpolation):
        if len(sample) == 2:
            image, targets = sample  # TARGETS ARE IN LABEL_CXCYWH
            with_crowd = False
        elif len(sample) == 3:
            image, targets, crowd_targets = sample
            with_crowd = True
        else:
            raise DatasetItemsException(data_sample=sample, collate_type=type(self), expected_item_names=self.expected_item_names)

        dsize = int(target_size), int(target_size)
        scale_factors = target_size / image.shape[0], target_size / image.shape[1]

        image = cv2.resize(
            image,
            dsize=dsize,
            interpolation=interpolation,
        )

        sy, sx = scale_factors
        targets[:, 1:5] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
        if with_crowd:
            crowd_targets[:, 1:5] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
            return image, targets, crowd_targets

        return image, targets


@register_collate_function()
class CrowdDetectionPPYoloECollateFN(PPYoloECollateFN):
    """
    Collate function for Yolox training with additional_batch_items that includes crowd targets
    """

    def __init__(self, random_resize_sizes: Union[List[int], None] = None, random_resize_modes: Union[List[int], None] = None):
        super().__init__(random_resize_sizes, random_resize_modes)
        self.expected_item_names = ("image", "targets", "crowd_targets")

    def __call__(self, data) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        if self.random_resize_sizes is not None:
            data = self.random_resize(data)

        try:
            images_batch, labels_batch, crowd_labels_batch = list(zip(*data))
        except (ValueError, TypeError):
            raise DatasetItemsException(data_sample=data[0], collate_type=type(self), expected_item_names=self.expected_item_names)

        return self._format_images(images_batch), self._format_targets(labels_batch), {"crowd_targets": self._format_targets(crowd_labels_batch)}
