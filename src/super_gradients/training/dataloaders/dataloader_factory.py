from hydra import initialize, compose
import hydra
from hydra.core.global_hydra import GlobalHydra

import super_gradients
from torch.utils.data import BatchSampler, DataLoader
from super_gradients.training.utils import get_param
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from super_gradients.common.factories.samplers_factory import SamplersFactory
from super_gradients.training.utils.distributed_training_utils import wait_for_the_master, get_local_rank
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def _get_data_loader(config_name, dataset_cls, train, dataset_params={}, dataloader_params={}):
    GlobalHydra.instance().clear()
    with initialize(config_path="../../recipes"):
        # config is relative to a module
        cfg = compose(config_name=config_name)

        _process_dataset_params(cfg, dataset_params, train)

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = dataset_cls(**dataset_params)

        _process_dataloader_params(cfg, dataloader_params, dataset, train)

        dataloader = DataLoader(dataset=dataset, **dataloader_params)
        return dataloader


def _process_dataset_params(cfg, dataset_params, train):
    default_dataset_params = cfg.dataset_params.train_dataset_params if train else cfg.dataset_params.val_dataset_params
    default_dataset_params = hydra.utils.instantiate(default_dataset_params)
    for key, val in default_dataset_params.items():
        if key not in dataset_params.keys() or dataset_params[key] is None:
            dataset_params[key] = val


def _process_dataloader_params(cfg, dataloader_params, dataset, train):
    default_dataloader_params = cfg.dataset_params.train_dataloader_params if train else cfg.dataset_params.val_dataloader_params
    default_dataloader_params = hydra.utils.instantiate(default_dataloader_params)
    is_dist = super_gradients.is_distributed()

    if get_param(default_dataloader_params, "sampler") is not None:
        _instantiate_sampler(dataset, default_dataloader_params)
    elif is_dist:
        default_dataloader_params["sampler"] = {"DistributedSampler": {}}
        _instantiate_sampler(dataset, default_dataloader_params)

    if get_param(dataloader_params, "sampler") is not None:
        _instantiate_sampler(dataset, dataloader_params)

    _override_default_params_without_nones(dataloader_params, default_dataloader_params)
    if get_param(dataloader_params, "batch_sampler"):
        sampler = dataloader_params.pop("sampler")
        batch_size = dataloader_params.pop("batch_size")
        dataloader_params["batch_sampler"] = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)


def _override_default_params_without_nones(params, default_params):
    for key, val in default_params.items():
        if key not in params.keys() or params[key] is None:
            params[key] = val


def _instantiate_sampler(dataset, dataloader_params):
    sampler_name = list(dataloader_params["sampler"].keys())[0]
    dataloader_params["sampler"][sampler_name]["dataset"] = dataset
    dataloader_params["sampler"] = SamplersFactory().get(dataloader_params["sampler"])


def coco2017_train(dataset_params={}, dataloader_params={}):
    return _get_data_loader(config_name="coco2017_yolox",
                            dataset_cls=COCODetectionDataset,
                            train=True,
                            dataset_params=dataset_params,
                            dataloader_params=dataloader_params
                            )


def coco2017_val(dataset_params={}, dataloader_params={}):
    return _get_data_loader(config_name="coco2017_yolox",
                            dataset_cls=COCODetectionDataset,
                            train=False,
                            dataset_params=dataset_params,
                            dataloader_params=dataloader_params
                            )
