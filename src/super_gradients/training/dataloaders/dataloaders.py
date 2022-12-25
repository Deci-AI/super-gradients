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

from super_gradients.training.datasets.detection_datasets.pascal_voc_detection import (
    PascalVOCUnifiedDetectionTrainDataset,
    PascalVOCDetectionDataset,
)
from super_gradients.training.utils import get_param
from super_gradients.training.utils.hydra_utils import normalize_path
from super_gradients.training.datasets import ImageNetDataset
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from super_gradients.training.datasets.classification_datasets.cifar import (
    Cifar10,
    Cifar100,
)
from super_gradients.training.datasets.segmentation_datasets import (
    CityscapesDataset,
    CoCoSegmentationDataSet,
    PascalVOC2012SegmentationDataSet,
    PascalVOCAndAUGUnifiedDataset,
    SuperviselyPersonsDataset,
)
from super_gradients.common.factories.samplers_factory import SamplersFactory
from super_gradients.training.utils.distributed_training_utils import (
    wait_for_the_master,
    get_local_rank,
)
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.utils import override_default_params_without_nones
from super_gradients.common.factories.datasets_factory import DatasetsFactory

logger = get_logger(__name__)


def get_data_loader(config_name, dataset_cls, train, dataset_params=None, dataloader_params=None):
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
    if dataloader_params is None:
        dataloader_params = dict()
    if dataset_params is None:
        dataset_params = dict()

    GlobalHydra.instance().clear()
    sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")
    dataset_config = os.path.join("dataset_params", config_name)
    with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
        # config is relative to a module
        cfg = compose(config_name=normalize_path(dataset_config))

        dataset_params = _process_dataset_params(cfg, dataset_params, train)

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = dataset_cls(**dataset_params)
            if not hasattr(dataset, "dataset_params"):
                dataset.dataset_params = dataset_params

        dataloader_params = _process_dataloader_params(cfg, dataloader_params, dataset, train)

        dataloader = DataLoader(dataset=dataset, **dataloader_params)
        dataloader.dataloader_params = dataloader_params
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
    dataloader_params = _process_sampler_params(dataloader_params, dataset, default_dataloader_params)

    return dataloader_params


def _process_sampler_params(dataloader_params, dataset, default_dataloader_params):
    is_dist = super_gradients.is_distributed()
    dataloader_params = override_default_params_without_nones(dataloader_params, default_dataloader_params)
    if get_param(dataloader_params, "sampler") is not None:
        dataloader_params = _instantiate_sampler(dataset, dataloader_params)
    elif is_dist:
        dataloader_params["sampler"] = {"DistributedSampler": {}}
        dataloader_params = _instantiate_sampler(dataset, dataloader_params)
    if get_param(dataloader_params, "batch_sampler"):
        sampler = dataloader_params.pop("sampler")
        batch_size = dataloader_params.pop("batch_size")
        if "drop_last" in dataloader_params:
            drop_last = dataloader_params.pop("drop_last")
        else:
            drop_last = dataloader_params["drop_last"]
        dataloader_params["batch_sampler"] = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
    return dataloader_params


def _instantiate_sampler(dataset, dataloader_params):
    sampler_name = list(dataloader_params["sampler"].keys())[0]
    if "shuffle" in dataloader_params.keys():
        # SHUFFLE IS MUTUALLY EXCLUSIVE WITH SAMPLER ARG IN DATALOADER INIT
        dataloader_params["sampler"][sampler_name]["shuffle"] = dataloader_params.pop("shuffle")
    dataloader_params["sampler"][sampler_name]["dataset"] = dataset
    dataloader_params["sampler"] = SamplersFactory().get(dataloader_params["sampler"])
    return dataloader_params


def coco2017_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="coco_detection_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def coco2017_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="coco_detection_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def coco2017_train_yolox(dataset_params: Dict = None, dataloader_params: Dict = None):
    return coco2017_train(dataset_params, dataloader_params)


def coco2017_val_yolox(dataset_params: Dict = None, dataloader_params: Dict = None):
    return coco2017_val(dataset_params, dataloader_params)


def coco2017_train_ssd_lite_mobilenet_v2(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="coco_detection_ssd_lite_mobilenet_v2_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def coco2017_val_ssd_lite_mobilenet_v2(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="coco_detection_ssd_lite_mobilenet_v2_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def imagenet_train(dataset_params=None, dataloader_params=None, config_name="imagenet_dataset_params"):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def imagenet_val(dataset_params=None, dataloader_params=None, config_name="imagenet_dataset_params"):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def imagenet_efficientnet_train(dataset_params=None, dataloader_params=None):
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_efficientnet_dataset_params",
    )


def imagenet_efficientnet_val(dataset_params=None, dataloader_params=None):
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_efficientnet_dataset_params",
    )


def imagenet_mobilenetv2_train(dataset_params=None, dataloader_params=None):
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv2_dataset_params",
    )


def imagenet_mobilenetv2_val(dataset_params=None, dataloader_params=None):
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv2_dataset_params",
    )


def imagenet_mobilenetv3_train(dataset_params=None, dataloader_params=None):
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv3_dataset_params",
    )


def imagenet_mobilenetv3_val(dataset_params=None, dataloader_params=None):
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv3_dataset_params",
    )


def imagenet_regnetY_train(dataset_params=None, dataloader_params=None):
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_regnetY_dataset_params")


def imagenet_regnetY_val(dataset_params=None, dataloader_params=None):
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_regnetY_dataset_params")


def imagenet_resnet50_train(dataset_params=None, dataloader_params=None):
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_dataset_params",
    )


def imagenet_resnet50_val(dataset_params=None, dataloader_params=None):
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_dataset_params",
    )


def imagenet_resnet50_kd_train(dataset_params=None, dataloader_params=None):
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_kd_dataset_params",
    )


def imagenet_resnet50_kd_val(dataset_params=None, dataloader_params=None):
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_kd_dataset_params",
    )


def imagenet_vit_base_train(dataset_params=None, dataloader_params=None):
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_vit_base_dataset_params",
    )


def imagenet_vit_base_val(dataset_params=None, dataloader_params=None):
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_vit_base_dataset_params",
    )


def tiny_imagenet_train(
    dataset_params=None,
    dataloader_params=None,
    config_name="tiny_imagenet_dataset_params",
):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def tiny_imagenet_val(
    dataset_params=None,
    dataloader_params=None,
    config_name="tiny_imagenet_dataset_params",
):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cifar10_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cifar10_dataset_params",
        dataset_cls=Cifar10,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cifar10_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cifar10_dataset_params",
        dataset_cls=Cifar10,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cifar100_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cifar100_dataset_params",
        dataset_cls=Cifar100,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cifar100_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cifar100_dataset_params",
        dataset_cls=Cifar100,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def classification_test_dataloader(batch_size: int = 5, image_size: int = 32, dataset_size=None) -> DataLoader:
    dataset_size = dataset_size or batch_size
    images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((dataset_size)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def detection_test_dataloader(batch_size: int = 5, image_size: int = 320, dataset_size=None) -> DataLoader:
    dataset_size = dataset_size or batch_size
    images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
    ground_truth = torch.Tensor(np.zeros((dataset_size, 6)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def segmentation_test_dataloader(batch_size: int = 5, image_size: int = 512, dataset_size=None) -> DataLoader:
    dataset_size = dataset_size or batch_size
    images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((dataset_size, image_size, image_size)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def cityscapes_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_stdc_seg50_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_stdc_seg50_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_stdc_seg50_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_stdc_seg50_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_stdc_seg75_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_stdc_seg75_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_stdc_seg75_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_stdc_seg75_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_regseg48_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_regseg48_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_regseg48_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_regseg48_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_ddrnet_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_ddrnet_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def cityscapes_ddrnet_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="cityscapes_ddrnet_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def coco_segmentation_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="coco_segmentation_dataset_params",
        dataset_cls=CoCoSegmentationDataSet,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def coco_segmentation_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="coco_segmentation_dataset_params",
        dataset_cls=CoCoSegmentationDataSet,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def pascal_aug_segmentation_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="pascal_aug_segmentation_dataset_params",
        dataset_cls=PascalVOCAndAUGUnifiedDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def pascal_aug_segmentation_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return pascal_voc_segmentation_val(dataset_params=dataset_params, dataloader_params=dataloader_params)


def pascal_voc_segmentation_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="pascal_voc_segmentation_dataset_params",
        dataset_cls=PascalVOC2012SegmentationDataSet,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def pascal_voc_segmentation_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="pascal_voc_segmentation_dataset_params",
        dataset_cls=PascalVOC2012SegmentationDataSet,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def supervisely_persons_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="supervisely_persons_dataset_params",
        dataset_cls=SuperviselyPersonsDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def supervisely_persons_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="supervisely_persons_dataset_params",
        dataset_cls=SuperviselyPersonsDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def pascal_voc_detection_train(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="pascal_voc_detection_dataset_params",
        dataset_cls=PascalVOCUnifiedDetectionTrainDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def pascal_voc_detection_val(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="pascal_voc_detection_dataset_params",
        dataset_cls=PascalVOCDetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


ALL_DATALOADERS = {
    "coco2017_train": coco2017_train,
    "coco2017_val": coco2017_val,
    "coco2017_train_yolox": coco2017_train_yolox,
    "coco2017_val_yolox": coco2017_val_yolox,
    "coco2017_train_ssd_lite_mobilenet_v2": coco2017_train_ssd_lite_mobilenet_v2,
    "coco2017_val_ssd_lite_mobilenet_v2": coco2017_val_ssd_lite_mobilenet_v2,
    "imagenet_train": imagenet_train,
    "imagenet_val": imagenet_val,
    "imagenet_efficientnet_train": imagenet_efficientnet_train,
    "imagenet_efficientnet_val": imagenet_efficientnet_val,
    "imagenet_mobilenetv2_train": imagenet_mobilenetv2_train,
    "imagenet_mobilenetv2_val": imagenet_mobilenetv2_val,
    "imagenet_mobilenetv3_train": imagenet_mobilenetv3_train,
    "imagenet_mobilenetv3_val": imagenet_mobilenetv3_val,
    "imagenet_regnetY_train": imagenet_regnetY_train,
    "imagenet_regnetY_val": imagenet_regnetY_val,
    "imagenet_resnet50_train": imagenet_resnet50_train,
    "imagenet_resnet50_val": imagenet_resnet50_val,
    "imagenet_resnet50_kd_train": imagenet_resnet50_kd_train,
    "imagenet_resnet50_kd_val": imagenet_resnet50_kd_val,
    "imagenet_vit_base_train": imagenet_vit_base_train,
    "imagenet_vit_base_val": imagenet_vit_base_val,
    "tiny_imagenet_train": tiny_imagenet_train,
    "tiny_imagenet_val": tiny_imagenet_val,
    "cifar10_train": cifar10_train,
    "cifar10_val": cifar10_val,
    "cifar100_train": cifar100_train,
    "cifar100_val": cifar100_val,
    "cityscapes_train": cityscapes_train,
    "cityscapes_val": cityscapes_val,
    "cityscapes_stdc_seg50_train": cityscapes_stdc_seg50_train,
    "cityscapes_stdc_seg50_val": cityscapes_stdc_seg50_val,
    "cityscapes_stdc_seg75_train": cityscapes_stdc_seg75_train,
    "cityscapes_stdc_seg75_val": cityscapes_stdc_seg75_val,
    "cityscapes_regseg48_train": cityscapes_regseg48_train,
    "cityscapes_regseg48_val": cityscapes_regseg48_val,
    "cityscapes_ddrnet_train": cityscapes_ddrnet_train,
    "cityscapes_ddrnet_val": cityscapes_ddrnet_val,
    "coco_segmentation_train": coco_segmentation_train,
    "coco_segmentation_val": coco_segmentation_val,
    "pascal_aug_segmentation_train": pascal_aug_segmentation_train,
    "pascal_aug_segmentation_val": pascal_aug_segmentation_val,
    "pascal_voc_segmentation_train": pascal_voc_segmentation_train,
    "pascal_voc_segmentation_val": pascal_voc_segmentation_val,
    "supervisely_persons_train": supervisely_persons_train,
    "supervisely_persons_val": supervisely_persons_val,
    "pascal_voc_detection_train": pascal_voc_detection_train,
    "pascal_voc_detection_val": pascal_voc_detection_val,
}


def get(name: str = None, dataset_params: Dict = None, dataloader_params: Dict = None, dataset: torch.utils.data.Dataset = None) -> DataLoader:
    """
    Get DataLoader of the recipe-configured dataset defined by name in ALL_DATALOADERS.

    :param name: dataset name in ALL_DATALOADERS.
    :param dataset_params: dataset params that override the yaml configured defaults, then passed to the dataset_cls.__init__.
    :param dataloader_params: DataLoader params that override the yaml configured defaults, then passed to the DataLoader.__init__
    :param dataset: torch.utils.data.Dataset to be used instead of passing "name" (i.e for external dataset objects).
    :return: initialized DataLoader.
    """
    if dataset is not None:
        if name or dataset_params:
            raise ValueError("'name' and 'dataset_params' cannot be passed with initialized dataset.")

    dataset_str = get_param(dataloader_params, "dataset")

    if dataset_str:
        if name or dataset:
            raise ValueError("'name' and 'datasets' cannot be passed when 'dataset' arg dataloader_params is used as well.")
        if dataset_params is not None:
            dataset = DatasetsFactory().get(conf={dataset_str: dataset_params})
        else:
            dataset = DatasetsFactory().get(conf=dataset_str)
        _ = dataloader_params.pop("dataset")

    if dataset is not None:
        dataloader_params = _process_sampler_params(dataloader_params, dataset, {})
        dataloader = DataLoader(dataset=dataset, **dataloader_params)
    elif name not in ALL_DATALOADERS.keys():
        raise ValueError("Unsupported dataloader: " + str(name))
    else:
        dataloader_cls = ALL_DATALOADERS[name]
        dataloader = dataloader_cls(dataset_params=dataset_params, dataloader_params=dataloader_params)

    return dataloader
