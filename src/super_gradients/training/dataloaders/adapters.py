from abc import ABC, abstractmethod
from typing import Optional, Iterable

import torch

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


class BaseDataloaderAdapterFactory(ABC):
    """Factory class, responsible for adapting datasets/dataloaders to seamlessly work with SG format."""

    @classmethod
    def from_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        config: Optional[DataConfig] = None,
        config_path: Optional[str] = None,
        collate_fn: Optional[callable] = None,
        **dataloader_kwargs,
    ) -> torch.utils.data.DataLoader:
        """Wrap a DataLoader to adapt its output to fit SuperGradients format for the specific task.

        :param dataset:         Dataset to adapt.
        :param config:          Adapter configuration. Use this if you want to explicitly set some specific params of your dataset.
                                Mutually exclusive with `config_path`.
        :param config_path:     Adapter cache path. Use this if you want to load and/or save the adapter config from a local path.
                                Mutually exclusive with `config`.
        :param collate_fn:      Collate function to use. Use this if you .If None, the pytorch default collate function will be used.

        :return:                Adapted DataLoader.
        """

        dataloader = torch.utils.data.DataLoader(dataset=dataset, **dataloader_kwargs)

        # `AdapterCollateFNClass` depends on the tasks, but just represents the collate function adapter for that specific task.
        AdapterCollateFNClass = cls._get_collate_fn_class()
        adapter_collate = AdapterCollateFNClass(base_collate_fn=collate_fn, config=config, config_path=config_path)

        _maybe_setup_adapter(adapter=adapter_collate.adapter, data=dataset)
        dataloader.collate_fn = adapter_collate
        return dataloader

    @classmethod
    def from_dataloader(
        cls,
        dataloader: torch.utils.data.DataLoader,
        config: Optional[DataConfig] = None,
        config_path: Optional[str] = None,
    ) -> torch.utils.data.DataLoader:
        """Wrap a DataLoader to adapt its output to fit SuperGradients format for the specific task.

        :param dataloader:      DataLoader to adapt.
        :param config:          Adapter configuration. Use this if you want to explicitly set some specific params of your dataset.
                                Mutually exclusive with `config_path`.
        :param config_path:     Adapter cache path. Use this if you want to load and/or save the adapter config from a local path.
                                Mutually exclusive with `config`.

        :return:                Adapted DataLoader.
        """

        # `AdapterCollateFNClass` depends on the tasks, but just represents the collate function adapter for that specific task.
        AdapterCollateFNClass = cls._get_collate_fn_class()
        adapter_collate = AdapterCollateFNClass(base_collate_fn=dataloader.collate_fn, config=config, config_path=config_path)

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


class DetectionDataloaderAdapterFactory(BaseDataloaderAdapterFactory):
    """Factory class, responsible for adapting datasets/dataloaders to seamlessly work with SG YOLOX, YOLONAS and PPYOLOE"""

    @classmethod
    def from_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        config: Optional[DataConfig] = None,
        config_path: Optional[str] = None,
        **dataloader_kwargs,
    ) -> torch.utils.data.DataLoader:
        return super().from_dataset(
            dataset=dataset,
            config=config,
            config_path=config_path,
            collate_fn=DetectionCollateFN(),  #
            **dataloader_kwargs,
        )

    @classmethod
    def _get_collate_fn_class(cls) -> type:
        return DetectionDatasetAdapterCollateFN


class SegmentationDataloaderAdapterFactory(BaseDataloaderAdapterFactory):
    @classmethod
    def _get_collate_fn_class(cls) -> type:
        return SegmentationDatasetAdapterCollateFN


class ClassificationDataloaderAdapterFactory(BaseDataloaderAdapterFactory):
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
                f"`{adapter.__class__.__name__}` can be used with DDP ONLY IF the config was initialized BEFORE.\n"
                "   - If you already have a cache file from a previous run, please use it. "
                "This may be the case if you ran the code already without DDP, or if you've used DataGradients.\n"
                "   - Otherwise, please run your script WITHOUT DDP first until your dataloader is adapted. Then re-run the same script but with DDP."
            )
        _ = adapter.adapt(next(iter(data)))  # Run a dummy iteration to ensure all the questions are asked.
        adapter.data_config.dump_cache_file()
