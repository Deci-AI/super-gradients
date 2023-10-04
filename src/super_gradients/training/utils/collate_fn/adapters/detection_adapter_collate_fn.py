from typing import Optional, Callable, Iterable, List, Tuple, Sequence

import torch
from torch.utils.data.dataloader import default_collate

from data_gradients.dataset_adapters.config.data_config import DetectionDataConfig
from data_gradients.dataset_adapters.detection_adapter import DetectionDatasetAdapter
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.collate_functions_factory import CollateFunctionsFactory
from super_gradients.common.registry import register_collate_function
from super_gradients.training.utils.collate_fn import DetectionCollateFN
from super_gradients.training.utils.collate_fn.adapters.base_adapter_collate_fn import BaseDatasetAdapterCollateFN


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
        :param adapter_config:  Adapter configuration. Use this if you want to hard code some specificities about your dataset.
                                Mutually exclusive with `adapter_cache_path`.
        :param adapter_cache_path:     Adapter cache path. Use this if you want to load and/or save the adapter config from a local path.
                                Mutually exclusive with `adapter_config`.
        :param collate_fn:     Collate function to use. Use this if you .If None, the pytorch default collate function will be used.
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
            images, targets = self._adapt(sample)  # Will construct batch of 1
            images, targets = images[0], targets[0]  # Extract the sample
            targets[:, 1:] = xyxy2cxcywh(targets[:, 1:])
            adapted_samples.append((images, targets))
        return adapted_samples

    def _adapt_batch(self, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        from super_gradients.training.utils.detection_utils import xyxy2cxcywh

        images, targets = super()._adapt_batch(batch)
        targets = DetectionCollateFN._format_targets(targets)
        targets[:, 2:] = xyxy2cxcywh(targets[:, 2:])  # Adapter returns xyxy
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
