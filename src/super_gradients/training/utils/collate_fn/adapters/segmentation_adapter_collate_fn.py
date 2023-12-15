from typing import Optional, Callable, Iterable, Tuple

import torch

from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig
from data_gradients.dataset_adapters.segmentation_adapter import SegmentationDatasetAdapter
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.collate_functions_factory import CollateFunctionsFactory
from super_gradients.common.registry import register_collate_function
from super_gradients.training.utils.collate_fn.adapters.base_adapter_collate_fn import BaseDatasetAdapterCollateFN


@register_collate_function()
class SegmentationDatasetAdapterCollateFN(BaseDatasetAdapterCollateFN):
    """Segmentation Collate function that adapts an input data to SuperGradients format

    This is done by applying the adapter logic either before or after the original collate function,
    depending on whether the adapter was set up on a batch or a sample.

    Note that the original collate function (if any) will still be used, but will be wrapped into this class.
    """

    @resolve_param("base_collate_fn", CollateFunctionsFactory())
    def __init__(self, config: Optional[SegmentationDataConfig] = None, config_path: Optional[str] = None, base_collate_fn: Optional[Callable] = None):
        """
        :param config:          Adapter configuration. Use this if you want to hard code some specificities about your dataset.
                                Mutually exclusive with `config_path`.
        :param config_path:     Adapter cache path. Use this if you want to load and/or save the adapter config from a local path.
                                Mutually exclusive with `config`.
        :param base_collate_fn: Collate function to use. Use this if you .If None, the pytorch default collate function will be used.
        """
        if config and config_path:
            raise ValueError("`config` and `config_path` cannot be set at the same time.")
        elif config is None and config_path:
            adapter = SegmentationDatasetAdapter.from_cache(cache_path=config_path)
        elif config is not None and config_path is None:
            adapter = SegmentationDatasetAdapter(data_config=config)
        else:
            raise ValueError("Please either set `config` or `config_path`.")

        super().__init__(adapter=adapter, base_collate_fn=base_collate_fn or base_collate_fn)

    def __call__(self, samples: Iterable[SupportedDataType]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = super().__call__(samples=samples)
        return images, targets
