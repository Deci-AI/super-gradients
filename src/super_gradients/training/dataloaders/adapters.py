from abc import ABC, abstractmethod
from typing import Optional, Iterable

import torch
from torch.utils.data.dataloader import default_collate

from data_gradients.dataset_adapters.config.data_config import DataConfig
from data_gradients.dataset_adapters.base_adapter import BaseDatasetAdapter
from data_gradients.dataset_adapters.config.typing_utils import SupportedDataType

from super_gradients.training.utils.collate_fn import DetectionCollateFN
from super_gradients.training.utils.collate_fn.adapters import (
    BaseDatasetAdapterCollateFN,
    ClassificationDatasetAdapterCollateFN,
    DetectionDatasetAdapterCollateFN,
    SegmentationDatasetAdapterCollateFN,
)


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


class DetectionDataloaderAdapter(BaseDataloaderAdapter):
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


def maybe_setup_dataloader_adapter(dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
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
