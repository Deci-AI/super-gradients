from typing import Dict, Mapping

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, UnsupportedValueType, DictConfig, open_dict
from torch.utils.data import BatchSampler, DataLoader, TensorDataset, RandomSampler

import super_gradients
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_dataloader, ALL_DATALOADERS
from super_gradients.common.factories.collate_functions_factory import CollateFunctionsFactory
from super_gradients.common.factories.datasets_factory import DatasetsFactory
from super_gradients.common.factories.samplers_factory import SamplersFactory
from super_gradients.common.object_names import Dataloaders
from super_gradients.training.datasets import ImageNetDataset
from super_gradients.training.datasets.classification_datasets.cifar import (
    Cifar10,
    Cifar100,
)
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset, RoboflowDetectionDataset, YoloDarknetFormatDetectionDataset
from super_gradients.training.datasets.detection_datasets.pascal_voc_detection import (
    PascalVOCUnifiedDetectionTrainDataset,
    PascalVOCDetectionDataset,
)
from super_gradients.training.datasets.pose_estimation_datasets import COCOKeypointsDataset
from super_gradients.training.datasets.pose_estimation_datasets.rescoring_dataset import TrainRescoringDataset, ValTrainRescoringDataset
from super_gradients.training.datasets.samplers import RepeatAugSampler
from super_gradients.training.datasets.segmentation_datasets import (
    CityscapesDataset,
    CoCoSegmentationDataSet,
    PascalVOC2012SegmentationDataSet,
    PascalVOCAndAUGUnifiedDataset,
    SuperviselyPersonsDataset,
    MapillaryDataset,
)
from super_gradients.training.utils import get_param
from super_gradients.training.utils.distributed_training_utils import (
    wait_for_the_master,
    get_local_rank,
)
from super_gradients.training.utils.utils import override_default_params_without_nones
from super_gradients.common.environment.cfg_utils import load_dataset_params
import torch.distributed as dist


logger = get_logger(__name__)


def get_data_loader(config_name: str, dataset_cls: object, train: bool, dataset_params: Mapping = None, dataloader_params: Mapping = None) -> DataLoader:
    """
    Class for creating dataloaders for taking defaults from yaml files in src/super_gradients/recipes.

    :param config_name: yaml config filename of dataset_params in recipes (for example coco_detection_dataset_params).
    :param dataset_cls: torch dataset uninitialized class.
    :param train: controls whether to take
        cfg.train_dataloader_params or cfg.valid_dataloader_params as defaults for the dataset constructor
     and
        cfg.train_dataset_params or cfg.valid_dataset_params as defaults for DataLoader contructor.

    :param dataset_params: dataset params that override the yaml configured defaults, then passed to the dataset_cls.__init__.
    :param dataloader_params: DataLoader params that override the yaml configured defaults, then passed to the DataLoader.__init__
    :return: DataLoader
    """
    if dataloader_params is None:
        dataloader_params = dict()
    if dataset_params is None:
        dataset_params = dict()

    cfg = load_dataset_params(config_name=config_name)

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


def _process_dataset_params(cfg, dataset_params, train: bool):
    """
    Merge the default dataset config with the user-provided overrides.
    This function handles variable interpolation in the dataset config.

    :param cfg: Default dataset config
    :param dataset_params: User-provided overrides
    :param train: boolean flag indicating whether we are processing train or val dataset params
    :return: New dataset params (merged defaults and overrides, where overrides take precedence)
    """

    try:
        # No, we can't simplify the following lines to:
        # >>> default_dataset_params = cfg.train_dataset_params if train else cfg.val_dataset_params
        # >>> dataset_params = OmegaConf.merge(default_dataset_params, dataset_params)
        # >>> return hydra.utils.instantiate(dataset_params)
        # For some reason this breaks interpolation :shrug:
        if not isinstance(dataset_params, DictConfig):
            dataset_params = OmegaConf.create(dataset_params)
        if train:
            train_dataset_params = cfg.train_dataset_params
            with open_dict(train_dataset_params):
                train_dataset_params.merge_with(dataset_params)
            cfg.train_dataset_params = train_dataset_params
            return hydra.utils.instantiate(cfg.train_dataset_params)
        else:
            val_dataset_params = cfg.val_dataset_params
            with open_dict(val_dataset_params):
                val_dataset_params.merge_with(dataset_params)
            cfg.val_dataset_params = val_dataset_params
            return hydra.utils.instantiate(cfg.val_dataset_params)

    except UnsupportedValueType:
        # This is somewhat ugly fallback for the case when the user provides overrides for the dataset params
        # that contains non-primitive types (E.g instantiated transforms).
        # In this case interpolation is not possible so we just override the default params with the user-provided ones.
        default_dataset_params = hydra.utils.instantiate(cfg.train_dataset_params if train else cfg.val_dataset_params)
        for key, val in default_dataset_params.items():
            if key not in dataset_params.keys() or dataset_params[key] is None:
                dataset_params[key] = val
        return dataset_params


def _process_dataloader_params(cfg, dataloader_params, dataset, train):
    default_dataloader_params = cfg.train_dataloader_params if train else cfg.val_dataloader_params
    default_dataloader_params = hydra.utils.instantiate(default_dataloader_params)
    dataloader_params = _process_sampler_params(dataloader_params, dataset, default_dataloader_params)
    dataloader_params = _process_collate_fn_params(dataloader_params)

    # The following check is needed to gracefully handle the rare but possible case when the dataset length
    # is less than the number of workers. In this case DataLoader will crash.
    # So we clamp the number of workers to not exceed the dataset length.
    num_workers = get_param(dataloader_params, "num_workers")
    if num_workers is not None and num_workers > 0:
        num_workers = min(num_workers, len(dataset))
        dataloader_params["num_workers"] = num_workers

    return dataloader_params


def _process_collate_fn_params(dataloader_params):
    if get_param(dataloader_params, "collate_fn") is not None:
        dataloader_params["collate_fn"] = CollateFunctionsFactory().get(dataloader_params["collate_fn"])

    return dataloader_params


def _process_sampler_params(dataloader_params, dataset, default_dataloader_params):
    is_dist = super_gradients.is_distributed()
    dataloader_params = override_default_params_without_nones(dataloader_params, default_dataloader_params)
    if get_param(dataloader_params, "sampler") is not None:
        dataloader_params = _instantiate_sampler(dataset, dataloader_params)
    elif is_dist:
        dataloader_params["sampler"] = {"DistributedSampler": {}}
        dataloader_params = _instantiate_sampler(dataset, dataloader_params)
        if get_param(dataloader_params, "min_samples") is not None:
            min_samples = dataloader_params.pop("min_samples")
            if len(dataset) < min_samples:
                world_size = dist.get_world_size()
                num_repeats = min_samples / len(dataset)
                selected_ratio = world_size / num_repeats

                # WE SET selected_ratio = world_size / num_repeats SO THAT IN RepeatAugSampler THE NUMBER OF SAMPLES
                # PER EPOCH PER RANK WILL BE DETERMINED BY
                # int(math.ceil(len(self.dataset) / selected_ratio)) =  min_samples / world_size

                dataloader_params["sampler"] = RepeatAugSampler(dataset=dataset, num_repeats=num_repeats, selected_round=0, selected_ratio=selected_ratio)
    elif get_param(dataloader_params, "min_samples") is not None:
        min_samples = dataloader_params.pop("min_samples")
        if len(dataset) < min_samples:
            dataloader_params["sampler"] = RandomSampler(dataset, replacement=True, num_samples=min_samples)
            if "shuffle" in dataloader_params.keys():
                dataloader_params.pop("shuffle")
            logger.info(f"Using min_samples={min_samples}")
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


@register_dataloader(Dataloaders.COCO2017_TRAIN)
def coco2017_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_VAL)
def coco2017_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_TRAIN_YOLO_NAS)
def coco2017_train_yolo_nas(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_nas_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_VAL_YOLO_NAS)
def coco2017_val_yolo_nas(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_nas_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_TRAIN_PPYOLOE)
def coco2017_train_ppyoloe(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_ppyoloe_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_VAL_PPYOLOE)
def coco2017_val_ppyoloe(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_ppyoloe_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_TRAIN_YOLOX)
def coco2017_train_yolox(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return coco2017_train(dataset_params, dataloader_params)


@register_dataloader(Dataloaders.COCO2017_VAL_YOLOX)
def coco2017_val_yolox(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return coco2017_val(dataset_params, dataloader_params)


@register_dataloader(Dataloaders.COCO2017_TRAIN_SSD_LITE_MOBILENET_V2)
def coco2017_train_ssd_lite_mobilenet_v2(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_ssd_lite_mobilenet_v2_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_VAL_SSD_LITE_MOBILENET_V2)
def coco2017_val_ssd_lite_mobilenet_v2(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_ssd_lite_mobilenet_v2_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.ROBOFLOW_TRAIN_BASE)
def roboflow_train_yolox(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="roboflow_detection_dataset_params",
        dataset_cls=RoboflowDetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.ROBOFLOW_VAL_BASE)
def roboflow_val_yolox(dataset_params: Dict = None, dataloader_params: Dict = None):
    return get_data_loader(
        config_name="roboflow_detection_dataset_params",
        dataset_cls=RoboflowDetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO_DETECTION_YOLO_FORMAT_TRAIN)
def coco_detection_yolo_format_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_format_base_dataset_params",
        dataset_cls=YoloDarknetFormatDetectionDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO_DETECTION_YOLO_FORMAT_VAL)
def coco_detection_yolo_format_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_detection_yolo_format_base_dataset_params",
        dataset_cls=YoloDarknetFormatDetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.IMAGENET_TRAIN)
def imagenet_train(dataset_params: Dict = None, dataloader_params: Dict = None, config_name="imagenet_dataset_params"):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.IMAGENET_VAL)
def imagenet_val(dataset_params: Dict = None, dataloader_params: Dict = None, config_name="imagenet_dataset_params"):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.IMAGENET_EFFICIENTNET_TRAIN)
def imagenet_efficientnet_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_efficientnet_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_EFFICIENTNET_VAL)
def imagenet_efficientnet_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_efficientnet_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_MOBILENETV2_TRAIN)
def imagenet_mobilenetv2_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv2_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_MOBILENETV2_VAL)
def imagenet_mobilenetv2_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv2_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_MOBILENETV3_TRAIN)
def imagenet_mobilenetv3_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv3_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_MOBILENETV3_VAL)
def imagenet_mobilenetv3_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_mobilenetv3_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_REGNETY_TRAIN)
def imagenet_regnetY_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_train(dataset_params, dataloader_params, config_name="imagenet_regnetY_dataset_params")


@register_dataloader(Dataloaders.IMAGENET_REGNETY_VAL)
def imagenet_regnetY_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_val(dataset_params, dataloader_params, config_name="imagenet_regnetY_dataset_params")


@register_dataloader(Dataloaders.IMAGENET_RESNET50_TRAIN)
def imagenet_resnet50_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_RESNET50_VAL)
def imagenet_resnet50_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_RESNET50_KD_TRAIN)
def imagenet_resnet50_kd_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_kd_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_RESNET50_KD_VAL)
def imagenet_resnet50_kd_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_resnet50_kd_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_VIT_BASE_TRAIN)
def imagenet_vit_base_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_train(
        dataset_params,
        dataloader_params,
        config_name="imagenet_vit_base_dataset_params",
    )


@register_dataloader(Dataloaders.IMAGENET_VIT_BASE_VAL)
def imagenet_vit_base_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return imagenet_val(
        dataset_params,
        dataloader_params,
        config_name="imagenet_vit_base_dataset_params",
    )


@register_dataloader(Dataloaders.TINY_IMAGENET_TRAIN)
def tiny_imagenet_train(
    dataset_params: Dict = None,
    dataloader_params: Dict = None,
    config_name="tiny_imagenet_dataset_params",
):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.TINY_IMAGENET_VAL)
def tiny_imagenet_val(
    dataset_params: Dict = None,
    dataloader_params: Dict = None,
    config_name="tiny_imagenet_dataset_params",
):
    return get_data_loader(
        config_name=config_name,
        dataset_cls=ImageNetDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CIFAR10_TRAIN)
def cifar10_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cifar10_dataset_params",
        dataset_cls=Cifar10,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CIFAR10_VAL)
def cifar10_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cifar10_dataset_params",
        dataset_cls=Cifar10,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CIFAR100_TRAIN)
def cifar100_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cifar100_dataset_params",
        dataset_cls=Cifar100,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CIFAR100_VAL)
def cifar100_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cifar100_dataset_params",
        dataset_cls=Cifar100,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def classification_test_dataloader(batch_size: int = 5, image_size: int = 32, dataset_size: int = None) -> DataLoader:
    dataset_size = dataset_size or batch_size
    images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((dataset_size)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def detection_test_dataloader(batch_size: int = 5, image_size: int = 320, dataset_size: int = None) -> DataLoader:
    dataset_size = dataset_size or batch_size
    images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
    ground_truth = torch.Tensor(np.zeros((dataset_size, 6)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


def segmentation_test_dataloader(batch_size: int = 5, image_size: int = 512, dataset_size: int = None) -> DataLoader:
    dataset_size = dataset_size or batch_size
    images = torch.Tensor(np.zeros((dataset_size, 3, image_size, image_size)))
    ground_truth = torch.LongTensor(np.zeros((dataset_size, image_size, image_size)))
    dataset = TensorDataset(images, ground_truth)
    return DataLoader(dataset=dataset, batch_size=batch_size)


@register_dataloader(Dataloaders.CITYSCAPES_TRAIN)
def cityscapes_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_VAL)
def cityscapes_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_STDC_SEG50_TRAIN)
def cityscapes_stdc_seg50_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_stdc_seg50_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_STDC_SEG50_VAL)
def cityscapes_stdc_seg50_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_stdc_seg50_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_STDC_SEG75_TRAIN)
def cityscapes_stdc_seg75_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_stdc_seg75_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_STDC_SEG75_VAL)
def cityscapes_stdc_seg75_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_stdc_seg75_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_REGSEG48_TRAIN)
def cityscapes_regseg48_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_regseg48_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_REGSEG48_VAL)
def cityscapes_regseg48_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_regseg48_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_DDRNET_TRAIN)
def cityscapes_ddrnet_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_ddrnet_dataset_params",
        dataset_cls=CityscapesDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.CITYSCAPES_DDRNET_VAL)
def cityscapes_ddrnet_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="cityscapes_ddrnet_dataset_params",
        dataset_cls=CityscapesDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO_SEGMENTATION_TRAIN)
def coco_segmentation_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_segmentation_dataset_params",
        dataset_cls=CoCoSegmentationDataSet,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO_SEGMENTATION_VAL)
def coco_segmentation_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_segmentation_dataset_params",
        dataset_cls=CoCoSegmentationDataSet,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.PASCAL_AUG_SEGMENTATION_TRAIN)
def pascal_aug_segmentation_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="pascal_aug_segmentation_dataset_params",
        dataset_cls=PascalVOCAndAUGUnifiedDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.PASCAL_AUG_SEGMENTATION_VAL)
def pascal_aug_segmentation_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return pascal_voc_segmentation_val(dataset_params=dataset_params, dataloader_params=dataloader_params)


@register_dataloader(Dataloaders.PASCAL_VOC_SEGMENTATION_TRAIN)
def pascal_voc_segmentation_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="pascal_voc_segmentation_dataset_params",
        dataset_cls=PascalVOC2012SegmentationDataSet,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.PASCAL_VOC_SEGMENTATION_VAL)
def pascal_voc_segmentation_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="pascal_voc_segmentation_dataset_params",
        dataset_cls=PascalVOC2012SegmentationDataSet,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.SUPERVISELY_PERSONS_TRAIN)
def supervisely_persons_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="supervisely_persons_dataset_params",
        dataset_cls=SuperviselyPersonsDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.SUPERVISELY_PERSONS_VAL)
def supervisely_persons_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="supervisely_persons_dataset_params",
        dataset_cls=SuperviselyPersonsDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.MAPILLARY_TRAIN)
def mapillary_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="mapillary_dataset_params",
        dataset_cls=MapillaryDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.MAPILLARY_VAL)
def mapillary_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="mapillary_dataset_params",
        dataset_cls=MapillaryDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.PASCAL_VOC_DETECTION_TRAIN)
def pascal_voc_detection_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="pascal_voc_detection_dataset_params",
        dataset_cls=PascalVOCUnifiedDetectionTrainDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.PASCAL_VOC_DETECTION_VAL)
def pascal_voc_detection_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="pascal_voc_detection_dataset_params",
        dataset_cls=PascalVOCDetectionDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_POSE_TRAIN)
def coco2017_pose_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_pose_estimation_dataset_params",
        dataset_cls=COCOKeypointsDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_POSE_VAL)
def coco2017_pose_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_pose_estimation_dataset_params",
        dataset_cls=COCOKeypointsDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_RESCORING_TRAIN)
def coco2017_rescoring_train(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_pose_estimation_rescoring_dataset_params",
        dataset_cls=TrainRescoringDataset,
        train=True,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


@register_dataloader(Dataloaders.COCO2017_RESCORING_VAL)
def coco2017_rescoring_val(dataset_params: Dict = None, dataloader_params: Dict = None) -> DataLoader:
    return get_data_loader(
        config_name="coco_pose_estimation_rescoring_dataset_params",
        dataset_cls=ValTrainRescoringDataset,
        train=False,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


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
