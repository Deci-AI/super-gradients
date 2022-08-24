import os.path

import pkg_resources
from hydra import compose, initialize_config_dir
import hydra
from hydra.core.global_hydra import GlobalHydra
from typing import Dict

import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

import super_gradients

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory

from torch.utils.data import BatchSampler, DataLoader
from super_gradients.training.utils import get_param
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from super_gradients.common.factories.samplers_factory import SamplersFactory
from super_gradients.training.utils.distributed_training_utils import wait_for_the_master, get_local_rank
from super_gradients.common.abstractions.abstract_logger import get_logger

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

    if get_param(default_dataloader_params, "sampler") is not None:
        default_dataloader_params = _instantiate_sampler(dataset, default_dataloader_params)
    elif is_dist:
        default_dataloader_params["sampler"] = {"DistributedSampler": {}}
        default_dataloader_params = _instantiate_sampler(dataset, default_dataloader_params)

    if get_param(dataloader_params, "sampler") is not None:
        dataloader_params = _instantiate_sampler(dataset, dataloader_params)

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


<<<<<<< HEAD
def coco2017_val(dataset_params: Dict = {}, dataloader_params: Dict = {}):
    return get_data_loader(config_name="coco_detection_dataset_params",
                           dataset_cls=COCODetectionDataset,
                           train=False,
                           dataset_params=dataset_params,
                           dataloader_params=dataloader_params
                           )
=======
def coco2017_val(dataset_params={}, dataloader_params={}):
    return _get_data_loader(config_name="coco_detection_yolox_dataset_params",
                            dataset_cls=COCODetectionDataset,
                            train=False,
                            dataset_params=dataset_params,
                            dataloader_params=dataloader_params
                            )
>>>>>>> 55e97b4f (base working)

class ImageFolder(torch_datasets.ImageFolder):
    @resolve_param('transform', factory=TransformsFactory())
    def __init__(self, root: str, transform: torch_transforms.Compose = None, *args, **kwargs):
        super(ImageFolder, self).__init__(root, transform, *args, **kwargs)


def imagenet_train(dataset_params={}, dataloader_params={}):
    return _get_data_loader(config_name="imagenet_base",
                            dataset_cls=ImageFolder,
                            train=True,
                            dataset_params=dataset_params,
                            dataloader_params=dataloader_params
                            )


def imagenet_val(dataset_params={}, dataloader_params={}):
    return _get_data_loader(config_name="imagenet_base",
                            dataset_cls=ImageFolder,
                            train=False,
                            dataset_params=dataset_params,
                            dataloader_params=dataloader_params
                            )


# self.trainset = torch_datasets.ImageFolder(root=traindir, transform=transforms.Compose(train_transformation_list))
# self.valset = torch_datasets.ImageFolder(valdir, transforms.Compose([
#     transforms.Resize(resize_size),
#     transforms.CenterCrop(crop_size),
#     transforms.ToTensor(),
#     normalize,
# ]))