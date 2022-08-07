import os

import numpy as np

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset, BatchSampler, DataLoader
import torchvision.transforms as transforms


from super_gradients.common import DatasetDataInterface
from super_gradients.common.environment import AWS_ENV_NAME
from super_gradients.common.abstractions.abstract_logger import get_logger

from super_gradients.training import utils as core_utils
from super_gradients.training.utils.distributed_training_utils import get_local_rank, wait_for_the_master

from super_gradients.training.utils import get_param
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat

from super_gradients.training.datasets import datasets_utils, DataAugmentation
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.datasets.data_augmentation import Lighting, RandomErase
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset, PascalVOCDetectionDataset

from super_gradients.training.datasets.samplers.infinite_sampler import InfiniteSampler
from super_gradients.training.datasets.segmentation_datasets import PascalVOC2012SegmentationDataSet, \
    PascalAUG2012SegmentationDataSet, CoCoSegmentationDataSet
from super_gradients.training.datasets.segmentation_datasets.cityscape_segmentation import CityscapesDataset
from super_gradients.training.datasets.segmentation_datasets.supervisely_persons_segmentation import \
    SuperviselyPersonsDataset

from super_gradients.training.datasets.samplers.repeated_augmentation_sampler import RepeatAugSampler
from super_gradients.training.datasets.datasets_utils import RandomResizedCropAndInterpolation, worker_init_reset_seed

from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionMixup, DetectionRandomAffine,\
    DetectionTargetsFormatTransform, DetectionPaddedRescale, DetectionHSV, DetectionHorizontalFlip

from super_gradients.training.exceptions.dataset_exceptions import IllegalDatasetParameterException


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

            - `aug_repeat_count` : int (default=0)

                amount of repetitions (each repetition of an example is augmented differently) for each
                 example for the trainset.

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
        # CHANGE THE BATCH SIZE ACCORDING TO THE NUMBER OF DEVICES - ONLY IN NON-DISTRIBUTED TRAINING MODE
        # IN DISTRIBUTED MODE WE NEED DISTRIBUTED SAMPLERS
        # NO SHUFFLE IN DISTRIBUTED TRAINING

        aug_repeat_count = get_param(self.dataset_params, "aug_repeat_count", 0)
        if aug_repeat_count > 0 and not distributed_sampler:
            raise IllegalDatasetParameterException("repeated augmentation is only supported with DDP.")

        if distributed_sampler:
            self.batch_size_factor = 1
            train_sampler = RepeatAugSampler(self.trainset,
                                             num_repeats=aug_repeat_count) if aug_repeat_count > 0 else DistributedSampler(
                self.trainset)
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
        Get self.train_loader, self.val_loader, self.test_loader, self.classes.

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

        # OVERWRITE MEAN AND STD IF DEFINED IN DATASET PARAMS
        self.lib_dataset_params['mean'] = core_utils.get_param(self.dataset_params, 'img_mean',
                                                               default_val=self.lib_dataset_params['mean'])
        self.lib_dataset_params['std'] = core_utils.get_param(self.dataset_params, 'img_std',
                                                              default_val=self.lib_dataset_params['std'])

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
    def __init__(self, dataset_params={}, image_size=320, batch_size=4, classes=None):
        trainset = torch.utils.data.TensorDataset(torch.Tensor(np.zeros((batch_size, 3, image_size, image_size))),
                                                  torch.Tensor(np.zeros((batch_size, 6))))

        super(DetectionTestDatasetInterface, self).__init__(trainset=trainset, dataset_params=dataset_params,
                                                            classes=classes)


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
        img_mean = core_utils.get_param(self.dataset_params, 'img_mean', default_val=[0.485, 0.456, 0.406])
        img_std = core_utils.get_param(self.dataset_params, 'img_std', default_val=[0.229, 0.224, 0.225])
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

        img_mean = core_utils.get_param(self.dataset_params, 'img_mean', default_val=[0.4802, 0.4481, 0.3975])
        img_std = core_utils.get_param(self.dataset_params, 'img_std', default_val=[0.2770, 0.2691, 0.2821])
        normalize = transforms.Normalize(mean=img_mean,
                                         std=img_std)

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


class SuperviselyPersonsDatasetInterface(DatasetInterface):
    def __init__(self, dataset_params=None, cache_labels: bool = False, cache_images: bool = False):
        super().__init__(dataset_params=dataset_params)
        root_dir = get_param(dataset_params, "dataset_dir", "/data/supervisely-persons")

        self.trainset = SuperviselyPersonsDataset(
            root_dir=root_dir,
            list_file='train.csv',
            dataset_hyper_params=dataset_params,
            cache_labels=cache_labels,
            cache_images=cache_images,
            image_mask_transforms_aug=get_param(dataset_params, "image_mask_transforms_aug", transforms.Compose([])),
            augment=True
        )

        self.valset = SuperviselyPersonsDataset(
            root_dir=root_dir,
            list_file='val.csv',
            dataset_hyper_params=dataset_params,
            cache_labels=cache_labels,
            cache_images=cache_images,
            image_mask_transforms=get_param(dataset_params, "image_mask_transforms", transforms.Compose([])),
            augment=False
        )

        self.classes = self.trainset.classes


class DetectionDatasetInterface(DatasetInterface):
    def build_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, val_batch_size=None,
                           test_batch_size=None, distributed_sampler: bool = False):

        train_sampler = InfiniteSampler(len(self.trainset), seed=0)

        train_batch_sampler = BatchSampler(
            sampler=train_sampler,
            batch_size=self.dataset_params.batch_size,
            drop_last=False,
        )

        self.train_loader = DataLoader(self.trainset,
                                       batch_sampler=train_batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       worker_init_fn=worker_init_reset_seed,
                                       collate_fn=self.dataset_params.train_collate_fn)

        if distributed_sampler:
            sampler = torch.utils.data.distributed.DistributedSampler(self.valset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(self.valset)

        val_loader = torch.utils.data.DataLoader(self.valset,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 sampler=sampler,
                                                 batch_size=self.dataset_params.val_batch_size,
                                                 collate_fn=self.dataset_params.val_collate_fn)

        self.val_loader = val_loader


class PascalVOCUnifiedDetectionDatasetInterface(DetectionDatasetInterface):

    def __init__(self, dataset_params=None):
        if dataset_params is None:
            dataset_params = dict()
        super().__init__(dataset_params=dataset_params)

        self.data_dir = self.dataset_params.data_dir
        train_input_dim = (self.dataset_params.train_image_size, self.dataset_params.train_image_size)
        val_input_dim = (self.dataset_params.val_image_size, self.dataset_params.val_image_size)
        train_max_num_samples = get_param(self.dataset_params, "train_max_num_samples")
        val_max_num_samples = get_param(self.dataset_params, "val_max_num_samples")

        if self.dataset_params.download:
            PascalVOCDetectionDataset.download(data_dir=self.data_dir)

        train_dataset_names = ["train2007", "val2007", "train2012", "val2012"]
        # We divide train_max_num_samples between the datasets
        if train_max_num_samples:
            max_num_samples_per_train_dataset = [len(segment) for segment in np.array_split(range(train_max_num_samples), len(train_dataset_names))]
        else:
            max_num_samples_per_train_dataset = [None] * len(train_dataset_names)
        train_sets = [PascalVOCDetectionDataset(data_dir=self.data_dir,
                                                input_dim=train_input_dim,
                                                cache=self.dataset_params.cache_train_images,
                                                cache_path=self.dataset_params.cache_dir + "cache_train",
                                                transforms=self.dataset_params.train_transforms,
                                                images_sub_directory='images/' + trainset_name + '/',
                                                class_inclusion_list=self.dataset_params.class_inclusion_list,
                                                max_num_samples=max_num_samples_per_train_dataset[i])
                      for i, trainset_name in enumerate(train_dataset_names)]

        testset2007 = PascalVOCDetectionDataset(data_dir=self.data_dir,
                                                input_dim=val_input_dim,
                                                cache=self.dataset_params.cache_val_images,
                                                cache_path=self.dataset_params.cache_dir + "cache_valid",
                                                transforms=self.dataset_params.val_transforms,
                                                images_sub_directory='images/test2007/',
                                                class_inclusion_list=self.dataset_params.class_inclusion_list,
                                                max_num_samples=val_max_num_samples)

        self.classes = train_sets[1].classes
        self.trainset = ConcatDataset(train_sets)
        self.valset = testset2007

        self.trainset.collate_fn = self.dataset_params.train_collate_fn
        self.trainset.classes = self.classes
        self.trainset.img_size = self.dataset_params.train_image_size
        self.trainset.cache_labels = self.dataset_params.cache_train_images


class CoCoDetectionDatasetInterface(DetectionDatasetInterface):
    def __init__(self, dataset_params={}):
        super(CoCoDetectionDatasetInterface, self).__init__(dataset_params=dataset_params)

        train_input_dim = (self.dataset_params.train_image_size, self.dataset_params.train_image_size)
        targets_format = get_param(self.dataset_params, "targets_format", DetectionTargetsFormat.LABEL_CXCYWH)

        train_transforms = [DetectionMosaic(input_dim=train_input_dim,
                                            prob=self.dataset_params.mosaic_prob),
                            DetectionRandomAffine(degrees=self.dataset_params.degrees,
                                                  translate=self.dataset_params.translate,
                                                  scales=self.dataset_params.mosaic_scale,
                                                  shear=self.dataset_params.shear,
                                                  target_size=train_input_dim,
                                                  filter_box_candidates=self.dataset_params.filter_box_candidates,
                                                  wh_thr=self.dataset_params.wh_thr,
                                                  area_thr=self.dataset_params.area_thr,
                                                  ar_thr=self.dataset_params.ar_thr
                                                  ),
                            DetectionMixup(input_dim=train_input_dim,
                                           mixup_scale=self.dataset_params.mixup_scale,
                                           prob=self.dataset_params.mixup_prob,
                                           flip_prob=self.dataset_params.flip_prob),
                            DetectionHSV(prob=self.dataset_params.hsv_prob,
                                         hgain=self.dataset_params.hgain,
                                         sgain=self.dataset_params.sgain,
                                         vgain=self.dataset_params.vgain
                                         ),
                            DetectionHorizontalFlip(prob=self.dataset_params.flip_prob),
                            DetectionPaddedRescale(input_dim=train_input_dim, max_targets=120),
                            DetectionTargetsFormatTransform(output_format=targets_format)
                            ]

        # IF CACHE- CREATING THE CACHE FILE WILL HAPPEN ONLY FOR RANK 0, THEN ALL THE OTHER RANKS SIMPLY READ FROM IT.
        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            self.trainset = COCODetectionDataset(data_dir=self.dataset_params.data_dir,
                                                 name=self.dataset_params.train_subdir,
                                                 json_file=self.dataset_params.train_json_file,
                                                 img_size=train_input_dim,
                                                 cache=self.dataset_params.cache_train_images,
                                                 cache_dir_path=self.dataset_params.cache_dir_path,
                                                 transforms=train_transforms,
                                                 with_crowd=False)

        val_input_dim = (self.dataset_params.val_image_size, self.dataset_params.val_image_size)
        with_crowd = core_utils.get_param(self.dataset_params, 'with_crowd', default_val=True)

        # IF CACHE- CREATING THE CACHE FILE WILL HAPPEN ONLY FOR RANK 0, THEN ALL THE OTHER RANKS SIMPLY READ FROM IT.
        with wait_for_the_master(local_rank):
            self.valset = COCODetectionDataset(
                data_dir=self.dataset_params.data_dir,
                json_file=self.dataset_params.val_json_file,
                name=self.dataset_params.val_subdir,
                img_size=val_input_dim,
                transforms=[DetectionPaddedRescale(input_dim=val_input_dim),
                            DetectionTargetsFormatTransform(max_targets=50, output_format=targets_format)],
                cache=self.dataset_params.cache_val_images,
                cache_dir_path=self.dataset_params.cache_dir_path,
                with_crowd=with_crowd)
        self.classes = COCO_DETECTION_CLASSES_LIST
