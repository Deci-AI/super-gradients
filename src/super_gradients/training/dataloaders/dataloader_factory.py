import os.path
import pkg_resources
from typing import Dict

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, TensorDataset

import super_gradients
from super_gradients.training.utils import get_param
from super_gradients.training.datasets import ImageNetDataset
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from super_gradients.common.factories.samplers_factory import SamplersFactory
from super_gradients.training.utils.distributed_training_utils import wait_for_the_master, get_local_rank
from super_gradients.common.abstractions.abstract_logger import get_logger
from torchvision.datasets import CIFAR10, CIFAR100

logger = get_logger(__name__)


def get_data_loader(config_name, dataset_cls, train, dataset_params={}, dataloader_params={}):
    """
    Class for creating dataloaders for taking defaults from yaml files in src/super_gradients/recipes.

    :param config_name: yaml config filename in recipes (for example coco2017_yolox).
    :param dataset_cls: torch dataset uninitialized class.
    :param train: controls whether to take
        cfg.dataset_params.train_dataloader_params or cfg.dataset_params.valid_dataloader_params as defaults for the dataset constructor
     and
        cfg.dataset_params.train_dataset_params or cfg.dataset_params.valid_dataset_params as defaults for DataLoader contructor.

    :param dataset_params: dataset params that override the yaml configured defaults, then passed to the dataset_cls.__init__.
    :param dataloader_params: DataLoader params that override the yaml configured defaults, then passed to the DataLoader.__init__
    :return: DataLoader
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=pkg_resources.resource_filename("super_gradients.recipes", "")):
        # config is relative to a module
        cfg = compose(config_name=os.path.join("dataset_params", config_name))

        dataset_params = _process_dataset_params(cfg, dataset_params, train)

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = dataset_cls(**dataset_params)

        dataloader_params = _process_dataloader_params(cfg, dataloader_params, dataset, train)

        dataloader = DataLoader(dataset=dataset, **dataloader_params)
        return dataloader


def _process_dataset_params(cfg, dataset_params, train):
    default_dataset_params = cfg.dataset_params.train_dataset_params if train else cfg.dataset_params.val_dataset_params
    default_dataset_params = hydra.utils.instantiate(default_dataset_params)
    for key, val in default_dataset_params.items():
        if key not in dataset_params.keys() or dataset_params[key] is None:
            dataset_params[key] = val

    return dataset_params


def _process_dataloader_params(cfg, dataloader_params, dataset, train):
    default_dataloader_params = cfg.dataset_params.train_dataloader_params if train else cfg.dataset_params.val_dataloader_params
    default_dataloader_params = hydra.utils.instantiate(default_dataloader_params)
    is_dist = super_gradients.is_distributed()

    if get_param(dataloader_params, "sampler") is not None:
        dataloader_params = _instantiate_sampler(dataset, dataloader_params)
    elif get_param(default_dataloader_params, "sampler") is not None:
        default_dataloader_params = _instantiate_sampler(dataset, default_dataloader_params)
    elif is_dist:
        default_dataloader_params["sampler"] = {"DistributedSampler": {}}
        default_dataloader_params = _instantiate_sampler(dataset, default_dataloader_params)

    dataloader_params = _override_default_params_without_nones(dataloader_params, default_dataloader_params)
    if get_param(dataloader_params, "batch_sampler"):
        sampler = dataloader_params.pop("sampler")
        batch_size = dataloader_params.pop("batch_size")
        dataloader_params["batch_sampler"] = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

    return dataloader_params


def _override_default_params_without_nones(params, default_params):
    for key, val in default_params.items():
        if key not in params.keys() or params[key] is None:
            params[key] = val
    return params


def _instantiate_sampler(dataset, dataloader_params):
    sampler_name = list(dataloader_params["sampler"].keys())[0]
    dataloader_params["sampler"][sampler_name]["dataset"] = dataset
    dataloader_params["sampler"] = SamplersFactory().get(dataloader_params["sampler"])
    return dataloader_params


def coco2017_train(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="coco_detection_dataset_params",
                           dataset_cls=COCODetectionDataset,
                           train=True,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def coco2017_val(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="coco_detection_dataset_params",
                           dataset_cls=COCODetectionDataset,
                           train=False,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def coco2017_train_yolox(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return coco2017_train(dataset_params, dataloader_params)


def coco2017_val_yolox(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return coco2017_val(dataset_params, dataloader_params)


def coco2017_train_ssd_lite_mobilenet_v2(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="coco_detection_ssd_lite_mobilenet_v2_dataset_params",
                           dataset_cls=COCODetectionDataset,
                           train=True,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def coco2017_val_ssd_lite_mobilenet_v2(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="coco_detection_ssd_lite_mobilenet_v2_dataset_params",
                           dataset_cls=COCODetectionDataset,
                           train=False,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def imagenet_train(dataset_params={}, dataloader_params={}, config_name="imagenet_dataset_params"):
    return get_data_loader(config_name=config_name,
                           dataset_cls=ImageNetDataset,
                           train=True,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params)


def imagenet_val(dataset_params={}, dataloader_params={}, config_name="imagenet_dataset_params"):
    return get_data_loader(config_name=config_name,
                           dataset_cls=ImageNetDataset,
                           train=False,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params)


def imagenet_efficientnet_train(dataset_params={}, dataloader_params={}):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_efficientnet_dataset_params")


def imagenet_efficientnet_val(dataset_params={}, dataloader_params={}):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_efficientnet_dataset_params")


def imagenet_mobilenetv2_train(dataset_params={}, dataloader_params={}):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_mobilenetv2_dataset_params")


def imagenet_mobilenetv2_val(dataset_params={}, dataloader_params={}):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_mobilenetv2_dataset_params")


def imagenet_mobilenetv3_train(dataset_params={}, dataloader_params={}):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_mobilenetv3_dataset_params")


def imagenet_mobilenetv3_val(dataset_params={}, dataloader_params={}):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_mobilenetv3_dataset_params")


def imagenet_regnetY_train(dataset_params={}, dataloader_params={}):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_regnetY_dataset_params")


def imagenet_regnetY_val(dataset_params={}, dataloader_params={}):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_regnetY_dataset_params")


def imagenet_resnet50_train(dataset_params={}, dataloader_params={}):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_resnet50_dataset_params")


def imagenet_resnet50_val(dataset_params={}, dataloader_params={}):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_resnet50_dataset_params")


def imagenet_resnet50_kd_train(dataset_params={}, dataloader_params={}):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_resnet50_kd_dataset_params")


def imagenet_resnet50_kd_val(dataset_params={}, dataloader_params={}):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_resnet50_kd_dataset_params")


def imagenet_vit_base_train(dataset_params={}, dataloader_params={}):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_vit_base_dataset_params")


def imagenet_vit_base_val(dataset_params={}, dataloader_params={}):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_vit_base_dataset_params")


def tiny_imagenet_train(dataset_params={}, dataloader_params={}, config_name="tiny_imagenet_dataset_params"):
    return get_data_loader(config_name=config_name,
                           dataset_cls=ImageNetDataset,
                           train=True,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params)


def tiny_imagenet_val(dataset_params={}, dataloader_params={}, config_name="tiny_imagenet_dataset_params"):
    return get_data_loader(config_name=config_name,
                           dataset_cls=ImageNetDataset,
                           train=False,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params)


def cifar10_train(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="cifar10_dataset_params",
                           dataset_cls=CIFAR10,
                           train=True,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def cifar10_val(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="cifar10_dataset_params",
                           dataset_cls=CIFAR10,
                           train=False,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def cifar100_train(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="cifar100_dataset_params",
                           dataset_cls=CIFAR100,
                           train=True,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def cifar100_val(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="cifar100_dataset_params",
                           dataset_cls=CIFAR100,
                           train=False,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )


def classification_test_dataloader(batch_size: int = 5, image_size: int = 32) -> DataLoader:
    images = torch.Tensor(np.zeros((batch_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((batch_size)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def detection_test_dataloader(batch_size: int = 5, image_size: int = 320) -> DataLoader:
    images = torch.Tensor(np.zeros((batch_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((batch_size, 6)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def segmentation_test_dataloader(batch_size: int = 5, image_size: int = 512) -> DataLoader:
    images = torch.Tensor(np.zeros((batch_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((batch_size, image_size, image_size)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)

