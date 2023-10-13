from abc import ABC
from typing import Callable, Iterable, Tuple, List, Sequence
import torch
from torch.utils.data.dataloader import default_collate

from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.collate_functions_factory import CollateFunctionsFactory


class BaseDatasetAdapterCollateFN(ABC):
    """Base Collate function that adapts an input data to SuperGradients format

    This is done by applying the adapter logic either before or after the original collate function,
    depending on whether the adapter was set up on a batch or a sample.

    Note that the original collate function (if any) will still be used, but will be wrapped into this class.
    """

    @resolve_param("base_collate_fn", CollateFunctionsFactory())
    def __init__(self, adapter: BaseDatasetAdapter, base_collate_fn: Callable):
        """
        :param adapter:             Dataset adapter to use
        :param base_collate_fn:     Collate function to wrap. If None, the default collate function will be used.
        """
        self._adapt_on_batch = adapter.data_config.is_batch

        self.adapter = adapter
        self._base_collate_fn = base_collate_fn or default_collate

        if isinstance(self._base_collate_fn, type(self)):
            raise RuntimeError(f"You just tried to instantiate {self.__class__.__name__} with a `base_collate_fn` of the same type, which is not supported.")

    def __call__(self, samples: Iterable[SupportedDataType]) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._require_setup:
            # This is required because python `input` is no compatible multiprocessing (e.g. `num_workers > 0`, or `DDP`)
            # And if not `self._require_setup`, the adapter will need to ask at least one question using the python `input`
            raise RuntimeError(
                f"Trying to collate using `{self.__class__.__name__}`, but it was not fully set up yet. Please do one of the following\n"
                f"   - Call `{self.__class__.__name__}(...).setup_adapter(dataloader)` before iterating over the dataloader.\n"
                f"   - or Instantiate `{self.__class__.__name__}(config_path=...)` with `config_path` mapping to the cache file of "
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
            images, targets = self._adapt(data=sample)  # Will construct batch of 1
            images, targets = images.squeeze(0), targets.squeeze(0)  # Extract the sample
            adapted_samples.append((images, targets))
        return adapted_samples

    def _adapt_batch(self, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the adapter logic to a batch. This should be called only if the adapter was setup on a batch.
        :param batch: Batch of samples to adapt
        :return:      Adapted batch (Images, Targets)
        """
        return self._adapt(data=batch)

    def _adapt(self, data: Iterable[SupportedDataType]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = self.adapter.adapt(data)
        images = images.float()  # SG takes float as input
        return images, targets
