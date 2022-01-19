import os
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from super_gradients.common.abstractions.abstract_logger import get_logger
from torch.utils.data.distributed import DistributedSampler
from super_gradients.training.datasets import datasets_utils, DataAugmentation
from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation
from super_gradients.training.datasets.detection_datasets import COCODetectionDataSet, PascalVOCDetectionDataSet
from super_gradients.training.datasets.segmentation_datasets import PascalVOC2012SegmentationDataSet, \
    PascalAUG2012SegmentationDataSet, CoCoSegmentationDataSet
from super_gradients.training import utils as core_utils
from super_gradients.common import DatasetDataInterface
from super_gradients.common.environment import AWS_ENV_NAME
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.training.exceptions.dataset_exceptions import IllegalDatasetParameterException
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CityscapesDataset
from torch.utils.data import ConcatDataset

default_dataset_params = {"batch_size": 64, "val_batch_size": 200, "test_batch_size": 200, "dataset_dir": "./data/",
                          "s3_link": None}
LIBRARY_DATASETS = {
    "cifar10": {'class': datasets.CIFAR10, 'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
    "cifar100": {'class': datasets.CIFAR100, 'mean': (0.5071, 0.4865, 0.4409), 'std': (0.2673, 0.2564, 0.2762)},
    "SVHN": {'class': datasets.SVHN, 'mean': None, 'std': None}
}

logger = get_logger(__name__)


class DatasetInterface:
    """
    DatasetInterface - This class manages all of the "communiation" the Model has with the Data Sets
    """

    def __init__(self, dataset_params={}, train_loader=None, val_loader=None, test_loader=None, classes=None):
        """
        @param train_loader: torch.utils.data.Dataloader (optional) dataloader for training.
        @param test_loader: torch.utils.data.Dataloader (optional) dataloader for testing.
        @param classes: list of classes.

        Note: the above parameters will be discarded in case dataset_params is passed.

        @param dataset_params:

            - `batch_size` : int (default=64)

                Number of examples per batch for training. Large batch sizes are recommended.

            - `val_batch_size` : int (default=200)

                Number of examples per batch for validation. Large batch sizes are recommended.

            - `dataset_dir` : str (default="./data/")

                Directory location for the data. Data will be downloaded to this directory when getting it from a
                remote url.

            - `s3_link` : str (default=None)

                remote s3 link to download the data (optional).
        """

        self.dataset_params = core_utils.HpmStruct(**default_dataset_params)
        self.dataset_params.override(**dataset_params)

        self.trainset, self.valset, self.testset = None, None, None
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.classes = classes
        self.batch_size_factor = 1
        if self.dataset_params.s3_link is not None:
            self.download_from_cloud()

    def download_from_cloud(self):
        if self.dataset_params.s3_link is not None:
            env_name = AWS_ENV_NAME
            downloader = DatasetDataInterface(env=env_name)
            target_dir = self.dataset_params.dataset_dir
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            downloader.load_remote_dataset_file(self.dataset_params.s3_link, target_dir)

    def build_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, val_batch_size=None,
                           test_batch_size=None, distributed_sampler: bool = False):
        """

        define train, val (and optionally test) loaders. The method deals separately with distributed training and standard
        (non distributed, or parallel training). In the case of distributed training we need to rely on distributed
        samplers.
        :param batch_size_factor: int - factor to multiply the batch size (usually for multi gpu)
        :param num_workers: int - number of workers (parallel processes) for dataloaders
        :param train_batch_size: int - batch size for train loader, if None will be taken from dataset_params
        :param val_batch_size: int - batch size for val loader, if None will be taken from dataset_params
        :param distributed_sampler: boolean flag for distributed training mode
        :return: train_loader, val_loader, classes: list of classes
        """
        # CHANGE THE BATCH SIZE ACCORDING TO THE NUMBER OF DEVICES - ONLY IN NON-DISTRIBUED TRAINING MODE
        # IN DISTRIBUTED MODE WE NEED DISTRIBUTED SAMPLERS
        # NO SHUFFLE IN DISTRIBUTED TRAINING
        if distributed_sampler:
            self.batch_size_factor = 1
            train_sampler = DistributedSampler(self.trainset)
            val_sampler = DistributedSampler(self.valset)
            test_sampler = DistributedSampler(self.testset) if self.testset is not None else None
            train_shuffle = False
        else:
            self.batch_size_factor = batch_size_factor
            train_sampler = None
            val_sampler = None
            test_sampler = None
            train_shuffle = True

        if train_batch_size is None:
            train_batch_size = self.dataset_params.batch_size * self.batch_size_factor
        if val_batch_size is None:
            val_batch_size = self.dataset_params.val_batch_size * self.batch_size_factor
        if test_batch_size is None:
            test_batch_size = self.dataset_params.test_batch_size * self.batch_size_factor

        train_loader_drop_last = core_utils.get_param(self.dataset_params, 'train_loader_drop_last', default_val=False)

        cutmix = core_utils.get_param(self.dataset_params, 'cutmix', False)
        cutmix_params = core_utils.get_param(self.dataset_params, 'cutmix_params')

        # WRAPPING collate_fn
        train_collate_fn = core_utils.get_param(self.trainset, 'collate_fn')
        val_collate_fn = core_utils.get_param(self.valset, 'collate_fn')
        test_collate_fn = core_utils.get_param(self.testset, 'collate_fn')

        if cutmix and train_collate_fn is not None:
            raise IllegalDatasetParameterException("cutmix and collate function cannot be used together")

        if cutmix:
            # FIXME - cutmix should be available only in classification dataset. once we make sure all classification
            # datasets inherit from the same super class, we should move cutmix code to that class
            logger.warning("Cutmix/mixup was enabled. This feature is currently supported only "
                           "for classification datasets.")
            train_collate_fn = CollateMixup(**cutmix_params)

        # FIXME - UNDERSTAND IF THE num_replicas VARIBALE IS NEEDED
        # train_sampler = DistributedSampler(self.trainset,
        #                                    num_replicas=distributed_gpus_num) if distributed_sampler else None
        # val_sampler = DistributedSampler(self.valset,
        #                                   num_replicas=distributed_gpus_num) if distributed_sampler else None

        self.train_loader = torch.utils.data.DataLoader(self.trainset,
                                                        batch_size=train_batch_size,
                                                        shuffle=train_shuffle,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        sampler=train_sampler,
                                                        collate_fn=train_collate_fn,
                                                        drop_last=train_loader_drop_last)

        self.val_loader = torch.utils.data.DataLoader(self.valset,
                                                      batch_size=val_batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      pin_memory=True,
                                                      sampler=val_sampler,
                                                      collate_fn=val_collate_fn)

        if self.testset is not None:
            self.test_loader = torch.utils.data.DataLoader(self.testset,
                                                           batch_size=test_batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           sampler=test_sampler,
                                                           collate_fn=test_collate_fn)

        self.classes = self.trainset.classes

    def get_data_loaders(self, **kwargs):
        """
        Get self.train_loader, self.test_loader, self.classes.

        If the data loaders haven't been initialized yet, build them first.

        :param kwargs: kwargs are passed to build_data_loaders.

        """

        if self.train_loader is None and self.val_loader is None:
            self.build_data_loaders(**kwargs)

        return self.train_loader, self.val_loader, self.test_loader, self.classes

    def get_val_sample(self, num_samples=1):
        if num_samples > len(self.valset):
            raise Exception("Tried to load more samples than val-set size")
        if num_samples == 1:
            return self.valset[0]
        else:
            return self.valset[0:num_samples]

    def get_dataset_params(self):
        return self.dataset_params

    def print_dataset_details(self):
        logger.info("{} training samples, {} val samples, {} classes".format(len(self.trainset), len(self.valset),
                                                                             len(self.trainset.classes)))


class ExternalDatasetInterface(DatasetInterface):
    def __init__(self, train_loader, val_loader, num_classes, dataset_params={}):
        """
        ExternalDatasetInterface - A wrapper for external dataset interface that gets dataloaders from keras/TF
        and converts them to Torch-like dataloaders that return torch.Tensors after
        optional collate_fn while maintaining the same interface (connect_dataset_interface etc.)
            :train_loader:       The external train_loader
            :val_loader:        The external val_loader
            :num_classes:        The number of classes
            :dataset_params      The dict that includes the batch_size and/or the collate_fn

            :return:             DataLoaders that generate torch.Tensors batches after collate_fn
        """
        super().__init__(dataset_params)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classes = num_classes

    def get_data_loaders(self, batch_size_factor: int = 1, num_workers: int = 8, train_batch_size: int = None,
                         val_batch_size: int = None, distributed_sampler: bool = False):

        # CHANGE THE BATCH SIZE ACCORDING TO THE NUMBER OF DEVICES - ONLY IN NON-DISTRIBUED TRAINING MODE
        # IN DISTRIBUTED MODE WE NEED DISTRIBUTED SAMPLERS
        # NO SHUFFLE IN DISTRIBUTED TRAINING
        if distributed_sampler:
            self.batch_size_factor = 1
            train_sampler = DistributedSampler(self.trainset, shuffle=True)
            val_sampler = DistributedSampler(self.valset)
            train_shuffle = False
        else:
            self.batch_size_factor = batch_size_factor
            train_sampler = None
            val_sampler = None
            train_shuffle = True

        if train_batch_size is None:
            train_batch_size = self.dataset_params.batch_size * self.batch_size_factor
        if val_batch_size is None:
            val_batch_size = self.dataset_params.val_batch_size * self.batch_size_factor

        train_loader_drop_last = core_utils.get_param(self.dataset_params, 'train_loader_drop_last', default_val=False)

        # WRAPPING collate_fn
        train_collate_fn = core_utils.get_param(self.dataset_params, 'train_collate_fn')
        val_collate_fn = core_utils.get_param(self.dataset_params, 'val_collate_fn')

        # FIXME - UNDERSTAND IF THE num_replicas VARIBALE IS NEEDED
        # train_sampler = DistributedSampler(self.trainset,
        #                                    num_replicas=distributed_gpus_num) if distributed_sampler else None
        # val_sampler = DistributedSampler(self.valset,
        #                                   num_replicas=distributed_gpus_num) if distributed_sampler else None

        self.torch_train_loader = torch.utils.data.DataLoader(self.train_loader,
                                                              batch_size=train_batch_size,
                                                              shuffle=train_shuffle,
                                                              num_workers=num_workers,
                                                              pin_memory=True,
                                                              sampler=train_sampler,
                                                              collate_fn=train_collate_fn,
                                                              drop_last=train_loader_drop_last)

        self.torch_val_loader = torch.utils.data.DataLoader(self.val_loader,
                                                            batch_size=val_batch_size,
                                                            shuffle=False,
                                                            num_workers=num_workers,
                                                            pin_memory=True,
                                                            sampler=val_sampler,
                                                            collate_fn=val_collate_fn)

        return self.torch_train_loader, self.torch_val_loader, None, self.classes


class LibraryDatasetInterface(DatasetInterface):
    def __init__(self, name="cifar10", dataset_params={}, to_cutout=False):
        super(LibraryDatasetInterface, self).__init__(dataset_params)
        self.dataset_name = name
        if self.dataset_name not in LIBRARY_DATASETS.keys():
            raise Exception('dataset not found')
        self.lib_dataset_params = LIBRARY_DATASETS[self.dataset_name]

        if self.lib_dataset_params['mean'] is None:
            trainset = torchvision.datasets.SVHN(root=self.dataset_params.dataset_dir, split='train', download=True,
                                                 transform=transforms.ToTensor())
            self.lib_dataset_params['mean'], self.lib_dataset_params['std'] = datasets_utils.get_mean_and_std(trainset)

        crop_size = core_utils.get_param(self.dataset_params, 'crop_size', default_val=32)

        if to_cutout:
            transform_train = transforms.Compose([
                transforms.RandomCrop(crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
                DataAugmentation.normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
                DataAugmentation.cutout(16),
                DataAugmentation.to_tensor()
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
            ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])
        dataset_cls = self.lib_dataset_params["class"]
        self.trainset = dataset_cls(root=self.dataset_params.dataset_dir, train=True, download=True,
                                    transform=transform_train)

        self.valset = dataset_cls(root=self.dataset_params.dataset_dir, train=False, download=True,
                                  transform=transform_val)


class Cifar10DatasetInterface(LibraryDatasetInterface):
    def __init__(self, dataset_params={}):
        super(Cifar10DatasetInterface, self).__init__(name="cifar10", dataset_params=dataset_params)


class Cifar100DatasetInterface(LibraryDatasetInterface):
    def __init__(self, dataset_params={}):
        super(Cifar100DatasetInterface, self).__init__(name="cifar100", dataset_params=dataset_params)


class TestDatasetInterface(DatasetInterface):
    def __init__(self, trainset, dataset_params={}, classes=None):
        super(TestDatasetInterface, self).__init__(dataset_params)

        self.trainset = trainset
        self.valset = self.trainset
        self.testset = self.trainset
        self.classes = classes

    def get_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, val_batch_size=None,
                         distributed_sampler=False):
        self.trainset.classes = [0, 1, 2, 3, 4] if self.classes is None else self.classes
        return super().get_data_loaders(batch_size_factor=batch_size_factor,
                                        num_workers=num_workers,
                                        train_batch_size=train_batch_size,
                                        val_batch_size=val_batch_size,
                                        distributed_sampler=distributed_sampler)


class ClassificationTestDatasetInterface(TestDatasetInterface):
    def __init__(self, dataset_params={}, image_size=32, batch_size=5, classes=None):
        trainset = torch.utils.data.TensorDataset(torch.Tensor(np.zeros((batch_size, 3, image_size, image_size))),
                                                  torch.LongTensor(np.zeros((batch_size))))
        super(ClassificationTestDatasetInterface, self).__init__(trainset=trainset, dataset_params=dataset_params,
                                                                 classes=classes)


class SegmentationTestDatasetInterface(TestDatasetInterface):
    def __init__(self, dataset_params={}, image_size=512, batch_size=4):
        trainset = torch.utils.data.TensorDataset(torch.Tensor(np.zeros((batch_size, 3, image_size, image_size))),
                                                  torch.LongTensor(np.zeros((batch_size, image_size, image_size))))

        super(SegmentationTestDatasetInterface, self).__init__(trainset=trainset, dataset_params=dataset_params)


class DetectionTestDatasetInterface(TestDatasetInterface):
    def __init__(self, dataset_params={}, image_size=320, batch_size=4):
        trainset = torch.utils.data.TensorDataset(torch.Tensor(np.zeros((batch_size, 3, image_size, image_size))),
                                                  torch.Tensor(np.zeros((batch_size, 6))))

        super(DetectionTestDatasetInterface, self).__init__(trainset=trainset, dataset_params=dataset_params)


class TestYoloDetectionDatasetInterface(DatasetInterface):
    """
    note: the output size is (batch_size, 6) in the test while in real training
    the size of axis 0 can vary (the number of bounding boxes)
    """

    def __init__(self, dataset_params={}, input_dims=(3, 32, 32), batch_size=5):
        super().__init__(dataset_params)
        self.trainset = torch.utils.data.TensorDataset(torch.ones((batch_size, *input_dims)),
                                                       torch.ones((batch_size, 6)))
        self.trainset.classes = [0, 1, 2, 3, 4]
        self.valset = self.trainset


class ImageNetDatasetInterface(DatasetInterface):
    def __init__(self, dataset_params={}, data_dir="/data/Imagenet"):
        super(ImageNetDatasetInterface, self).__init__(dataset_params)

        data_dir = dataset_params['dataset_dir'] if 'dataset_dir' in dataset_params.keys() else data_dir
        traindir = os.path.join(os.path.abspath(data_dir), 'train')
        valdir = os.path.join(data_dir, 'val')
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=img_mean,
                                         std=img_std)

        crop_size = core_utils.get_param(self.dataset_params, 'crop_size', default_val=224)
        resize_size = core_utils.get_param(self.dataset_params, 'resize_size', default_val=256)
        color_jitter = core_utils.get_param(self.dataset_params, 'color_jitter', default_val=0.0)
        imagenet_pca_aug = core_utils.get_param(self.dataset_params, 'imagenet_pca_aug', default_val=0.0)
        train_interpolation = core_utils.get_param(self.dataset_params, 'train_interpolation', default_val='default')
        rand_augment_config_string = core_utils.get_param(self.dataset_params, 'rand_augment_config_string',
                                                          default_val=None)

        color_jitter = (float(color_jitter),) * 3 if isinstance(color_jitter, float) else color_jitter
        assert len(color_jitter) in (3, 4), "color_jitter must be a scalar or tuple of len 3 or 4"

        color_augmentation = datasets_utils.get_color_augmentation(rand_augment_config_string, color_jitter,
                                                                   crop_size=crop_size, img_mean=img_mean)

        train_transformation_list = [
            RandomResizedCropAndInterpolation(crop_size, interpolation=train_interpolation),
            transforms.RandomHorizontalFlip(),
            color_augmentation,
            transforms.ToTensor(),
            Lighting(imagenet_pca_aug),
            normalize]

        rndm_erase_prob = core_utils.get_param(self.dataset_params, 'random_erase_prob', default_val=0.)
        if rndm_erase_prob:
            train_transformation_list.append(RandomErase(rndm_erase_prob, self.dataset_params.random_erase_value))

        self.trainset = datasets.ImageFolder(traindir, transforms.Compose(train_transformation_list))
        self.valset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]))


class TinyImageNetDatasetInterface(DatasetInterface):
    def __init__(self, dataset_params={}, data_dir="/data/TinyImagenet"):
        super(TinyImageNetDatasetInterface, self).__init__(dataset_params)

        data_dir = dataset_params['dataset_dir'] if 'dataset_dir' in dataset_params.keys() else data_dir
        traindir = os.path.join(os.path.abspath(data_dir), 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                         std=[0.2770, 0.2691, 0.2821])

        crop_size = core_utils.get_param(self.dataset_params, 'crop_size', default_val=56)
        resize_size = core_utils.get_param(self.dataset_params, 'resize_size', default_val=64)

        self.trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        self.valset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]))


class ClassificationDatasetInterface(DatasetInterface):
    def __init__(self, normalization_mean=(0, 0, 0), normalization_std=(1, 1, 1), resolution=64,
                 dataset_params={}):
        super(ClassificationDatasetInterface, self).__init__(dataset_params)
        data_dir = self.dataset_params.dataset_dir

        traindir = os.path.join(os.path.abspath(data_dir), 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=normalization_mean,
                                         std=normalization_std)

        self.trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        self.valset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(resolution * 1.15)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ]))
        self.data_dir = data_dir
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std


class PascalVOC2012SegmentationDataSetInterface(DatasetInterface):
    def __init__(self, dataset_params=None, cache_labels=False, cache_images=False):
        if dataset_params is None:
            dataset_params = dict()
        super().__init__(dataset_params=dataset_params)

        self.root_dir = dataset_params['dataset_dir'] if 'dataset_dir' in dataset_params.keys() \
            else '/data/pascal_voc_2012/VOCdevkit/VOC2012/'

        self.trainset = PascalVOC2012SegmentationDataSet(root=self.root_dir,
                                                         list_file='ImageSets/Segmentation/train.txt',
                                                         samples_sub_directory='JPEGImages',
                                                         targets_sub_directory='SegmentationClass', augment=True,
                                                         dataset_hyper_params=dataset_params, cache_labels=cache_labels,
                                                         cache_images=cache_images)

        self.valset = PascalVOC2012SegmentationDataSet(root=self.root_dir,
                                                       list_file='ImageSets/Segmentation/val.txt',
                                                       samples_sub_directory='JPEGImages',
                                                       targets_sub_directory='SegmentationClass', augment=True,
                                                       dataset_hyper_params=dataset_params, cache_labels=cache_labels,
                                                       cache_images=cache_images)
        self.classes = self.trainset.classes


class PascalAUG2012SegmentationDataSetInterface(DatasetInterface):
    def __init__(self, dataset_params=None, cache_labels=False, cache_images=False):
        if dataset_params is None:
            dataset_params = dict()
        super().__init__(dataset_params=dataset_params)

        self.root_dir = dataset_params['dataset_dir'] if 'dataset_dir' in dataset_params.keys() \
            else '/data/pascal_voc_2012/VOCaug/dataset/'

        self.trainset = PascalAUG2012SegmentationDataSet(
            root=self.root_dir,
            list_file='trainval.txt',
            samples_sub_directory='img',
            targets_sub_directory='cls', augment=True,
            dataset_hyper_params=dataset_params, cache_labels=cache_labels,
            cache_images=cache_images)

        self.valset = PascalAUG2012SegmentationDataSet(
            root=self.root_dir,
            list_file='val.txt',
            samples_sub_directory='img',
            targets_sub_directory='cls', augment=False,
            dataset_hyper_params=dataset_params, cache_labels=cache_labels,
            cache_images=cache_images)

        self.classes = self.trainset.classes


class CoCoDataSetInterfaceBase(DatasetInterface):
    def __init__(self, dataset_params=None):
        if dataset_params is None:
            dataset_params = dict()
        super().__init__(dataset_params=dataset_params)

        self.root_dir = dataset_params['dataset_dir'] if 'dataset_dir' in dataset_params.keys() else '/data/coco/'


class CoCoDetectionDatasetInterface(CoCoDataSetInterfaceBase):
    def __init__(self, dataset_params=None, cache_labels=False, cache_images=False, train_list_file='train2017.txt',
                 val_list_file='val2017.txt'):
        super().__init__(dataset_params=dataset_params)

        default_hyper_params = {
            'hsv_h': 0.0138,  # IMAGE HSV-Hue AUGMENTATION (fraction)
            'hsv_s': 0.678,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
            'hsv_v': 0.36,  # IMAGE HSV-Value AUGMENTATION (fraction)
            'degrees': 1.98,  # IMAGE ROTATION (+/- deg)
            'translate': 0.05,  # IMAGE TRANSLATION (+/- fraction)
            'scale': 0.05,  # IMAGE SCALE (+/- gain)
            'shear': 0.641,  # IMAGE SHEAR (+/- deg)
            'mixup': 0.0}  # IMAGE MIXUP AUGMENTATION (fraction)

        self.coco_dataset_hyper_params = core_utils.get_param(self.dataset_params, 'dataset_hyper_param',
                                                              default_val=default_hyper_params)
        train_sample_method = core_utils.get_param(self.dataset_params, 'train_sample_loading_method',
                                                   default_val='mosaic')
        val_sample_method = core_utils.get_param(self.dataset_params, 'val_sample_loading_method',
                                                 default_val='rectangular')
        train_collate_fn = core_utils.get_param(self.dataset_params, 'train_collate_fn', base_detection_collate_fn)
        val_collate_fn = core_utils.get_param(self.dataset_params, 'val_collate_fn', base_detection_collate_fn)

        image_size = core_utils.get_param(self.dataset_params, 'image_size')
        train_image_size = core_utils.get_param(self.dataset_params, 'train_image_size')
        val_image_size = core_utils.get_param(self.dataset_params, 'val_image_size')
        labels_offset = core_utils.get_param(self.dataset_params, 'labels_offset', default_val=0)
        class_inclusion_list = core_utils.get_param(self.dataset_params, 'class_inclusion_list')

        if image_size is None:
            assert train_image_size is not None and val_image_size is not None, 'Please provide either only image_size or ' \
                                                                                'both train_image_size AND val_image_size'
        else:
            assert train_image_size is None and val_image_size is None, 'Please provide either only image_size or ' \
                                                                        'both train_image_size AND val_image_size'
            train_image_size = image_size
            val_image_size = image_size

        self.trainset = COCODetectionDataSet(root=self.root_dir, list_file=train_list_file,
                                             dataset_hyper_params=self.coco_dataset_hyper_params,
                                             batch_size=self.dataset_params.batch_size,
                                             img_size=train_image_size,
                                             collate_fn=train_collate_fn,
                                             augment=True,
                                             sample_loading_method=train_sample_method,
                                             cache_labels=cache_labels,
                                             cache_images=cache_images,
                                             labels_offset=labels_offset,
                                             class_inclusion_list=class_inclusion_list)

        self.valset = COCODetectionDataSet(root=self.root_dir, list_file=val_list_file,
                                           dataset_hyper_params=self.coco_dataset_hyper_params,
                                           batch_size=self.dataset_params.val_batch_size,
                                           img_size=val_image_size,
                                           collate_fn=val_collate_fn,
                                           sample_loading_method=val_sample_method,
                                           cache_labels=cache_labels,
                                           cache_images=cache_images,
                                           labels_offset=labels_offset,
                                           class_inclusion_list=class_inclusion_list)

        self.coco_classes = self.trainset.classes


class CoCoSegmentationDatasetInterface(CoCoDataSetInterfaceBase):
    def __init__(self, dataset_params=None, cache_labels: bool = False, cache_images: bool = False,
                 dataset_classes_inclusion_tuples_list: list = None):
        super().__init__(dataset_params=dataset_params)

        self.trainset = CoCoSegmentationDataSet(
            root=self.root_dir,
            list_file='instances_train2017.json',
            samples_sub_directory='images/train2017',
            targets_sub_directory='annotations', augment=True,
            dataset_hyper_params=dataset_params,
            cache_labels=cache_labels,
            cache_images=cache_images,
            dataset_classes_inclusion_tuples_list=dataset_classes_inclusion_tuples_list)

        self.valset = CoCoSegmentationDataSet(
            root=self.root_dir,
            list_file='instances_val2017.json',
            samples_sub_directory='images/val2017',
            targets_sub_directory='annotations', augment=False,
            dataset_hyper_params=dataset_params,
            cache_labels=cache_labels,
            cache_images=cache_images,
            dataset_classes_inclusion_tuples_list=dataset_classes_inclusion_tuples_list)

        self.coco_classes = self.trainset.classes


class CoCo2014DetectionDatasetInterface(CoCoDetectionDatasetInterface):
    def __init__(self, dataset_params=None, cache_labels=False, cache_images=False, train_list_file='train2014.txt',
                 val_list_file='val2014.txt'):
        dataset_params['dataset_dir'] = core_utils.get_param(dataset_params, 'dataset_dir', '/data/coco2014/')
        super().__init__(dataset_params=dataset_params, cache_labels=cache_labels, cache_images=cache_images,
                         train_list_file=train_list_file, val_list_file=val_list_file)


class CityscapesDatasetInterface(DatasetInterface):
    def __init__(self, dataset_params=None, cache_labels: bool = False, cache_images: bool = False):
        super().__init__(dataset_params=dataset_params)
        root_dir = core_utils.get_param(dataset_params, "dataset_dir", "/data/cityscapes")
        img_size = core_utils.get_param(dataset_params, "img_size", 1024)
        crop_size = core_utils.get_param(dataset_params, "crop_size", 512)
        image_mask_transforms = core_utils.get_param(dataset_params, "image_mask_transforms")
        image_mask_transforms_aug = core_utils.get_param(dataset_params, "image_mask_transforms_aug")

        self.trainset = CityscapesDataset(
            root_dir=root_dir,
            list_file='lists/train.lst',
            labels_csv_path="lists/labels.csv",
            img_size=img_size,
            crop_size=crop_size,
            augment=True,
            dataset_hyper_params=dataset_params,
            cache_labels=cache_labels,
            cache_images=cache_images,
            image_mask_transforms=image_mask_transforms,
            image_mask_transforms_aug=image_mask_transforms_aug)

        self.valset = CityscapesDataset(
            root_dir=root_dir,
            list_file='lists/val.lst',
            labels_csv_path="lists/labels.csv",
            img_size=img_size,
            crop_size=crop_size,
            augment=False,
            dataset_hyper_params=dataset_params,
            cache_labels=cache_labels,
            cache_images=cache_images,
            image_mask_transforms=image_mask_transforms)

        self.classes = self.trainset.classes


class PascalVOCUnifiedDetectionDataSetInterface(DatasetInterface):
    def __init__(self, dataset_params=None, cache_labels=False, cache_images=False):
        if dataset_params is None:
            dataset_params = dict()
        super().__init__(dataset_params=dataset_params)

        default_hyper_params = {
            'hsv_h': 0.0138,  # IMAGE HSV-Hue AUGMENTATION (fraction)
            'hsv_s': 0.664,  # IMAGE HSV-Saturation AUGMENTATION (fraction)
            'hsv_v': 0.464,  # IMAGE HSV-Value AUGMENTATION (fraction)
            'degrees': 0.373,  # IMAGE ROTATION (+/- deg)
            'translate': 0.245,  # IMAGE TRANSLATION (+/- fraction)
            'scale': 0.898,  # IMAGE SCALE (+/- gain)
            'shear': 0.602}  # IMAGE SHEAR (+/- deg)

        self.pascal_voc_dataset_hyper_params = core_utils.get_param(self.dataset_params, 'dataset_hyper_param',
                                                                    default_val=default_hyper_params)
        train_sample_method = core_utils.get_param(self.dataset_params, 'train_sample_loading_method',
                                                   default_val='mosaic')
        val_sample_method = core_utils.get_param(self.dataset_params, 'val_sample_loading_method',
                                                 default_val='rectangular')
        train_collate_fn = core_utils.get_param(self.dataset_params, 'train_collate_fn')
        val_collate_fn = core_utils.get_param(self.dataset_params, 'val_collate_fn')

        train_sets = []
        for trainset_prefix in ["train", "val"]:
            for trainset_year in ["2007", "2012"]:
                sub_trainset = PascalVOCDetectionDataSet(root="~/data/pascal_unified_coco_format/",
                                                         list_file='images/VOCdevkit/VOC'+trainset_year+'/ImageSets/Main/train.txt',
                                                         samples_sub_directory='images/'+trainset_prefix+trainset_year+'/',
                                                         targets_sub_directory='labels/'+trainset_prefix+trainset_year,
                                                         dataset_hyper_params=self.pascal_voc_dataset_hyper_params,
                                                         batch_size=self.dataset_params.batch_size,
                                                         img_size=self.dataset_params.train_image_size,
                                                         collate_fn=train_collate_fn,
                                                         sample_loading_method=train_sample_method,
                                                         cache_labels=cache_labels,
                                                         cache_images=cache_images,
                                                         augment=True)
                train_sets.append(sub_trainset)

        testset2007 = PascalVOCDetectionDataSet(root="~/data/pascal_unified_coco_format/",
                                                list_file='images/VOCdevkit/VOC2007/ImageSets/Main/test.txt',
                                                samples_sub_directory='images/test2007/',
                                                targets_sub_directory='labels/test2007',
                                                dataset_hyper_params=self.pascal_voc_dataset_hyper_params,
                                                batch_size=self.dataset_params.val_batch_size,
                                                img_size=self.dataset_params.val_image_size,
                                                collate_fn=val_collate_fn,
                                                sample_loading_method=val_sample_method,
                                                cache_labels=cache_labels,
                                                cache_images=cache_images,
                                                augment=False)

        self.classes = train_sets[1].classes
        self.trainset = ConcatDataset(train_sets)
        self.trainset.collate_fn = train_collate_fn
        self.valset = testset2007
        self.trainset.classes = self.classes
