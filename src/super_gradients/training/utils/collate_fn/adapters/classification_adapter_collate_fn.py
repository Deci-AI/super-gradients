from typing import Optional, Callable, Iterable, Tuple

import torch

from data_gradients.dataset_adapters.config.data_config import ClassificationDataConfig
from data_gradients.dataset_adapters.classification_adapter import ClassificationDatasetAdapter
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.collate_functions_factory import CollateFunctionsFactory
from super_gradients.common.registry import register_collate_function
from super_gradients.training.utils.collate_fn.adapters.base_adapter_collate_fn import BaseDatasetAdapterCollateFN


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

        super().__init__(adapter=adapter, base_collate_fn=base_collate_fn or base_collate_fn)

    def __call__(self, samples: Iterable[SupportedDataType]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = super().__call__(samples=samples)  # This already returns a batch of (images, targets)
        images = images / 255
        return images, targets
