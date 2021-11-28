import os
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Callable
import torchvision.transforms as transform
from PIL import Image
from super_gradients.training.datasets.sg_dataset import DirectoryDataSet, ListDataset
from super_gradients.training.utils.segmentation_utils import RandomFlip, Rescale, RandomRotate, PadShortToCropSize,\
    CropImageAndMask, RandomGaussianBlur, RandomRescale


class SegmentationDataSet(DirectoryDataSet, ListDataset):
    def __init__(self, root: str, list_file: str = None, samples_sub_directory: str = None,
                 targets_sub_directory: str = None,
                 img_size: int = 608, crop_size: int = 512, batch_size: int = 16, augment: bool = False,
                 dataset_hyper_params: dict = None,
                 cache_labels: bool = False, cache_images: bool = False, sample_loader: Callable = None,
                 target_loader: Callable = None, collate_fn: Callable = None, target_extension: str = '.png',
                 image_mask_transforms: transform.Compose = None, image_mask_transforms_aug: transform.Compose = None):
        """
        SegmentationDataSet
                                * Please use self.augment == True only for training

            :param root:                        Root folder of the Data Set
            :param list_file:                   Path to the file with the samples list
            :param samples_sub_directory:       name of the samples sub-directory
            :param targets_sub_directory:       name of the targets sub-directory
            :param img_size:                    Image size of the Model that uses this Data Set
            :param crop_size:                   The size of the cropped image
            :param batch_size:                  Batch Size of the Model that uses this Data Set
            :param augment:                     True / False flag to allow Augmentation
            :param dataset_hyper_params:        Any hyper params required for the data set
            :param cache_labels:                "Caches" the labels -> Pre-Loads to memory as a list
            :param cache_images:                "Caches" the images -> Pre-Loads to memory as a list
            :param sample_loader:               A function that specifies how to load a sample
            :param target_loader:               A function that specifies how to load a target
            :param collate_fn:                  collate_fn func to process batches for the Data Loader
            :param target_extension:            file extension of the targets (defualt is .png for PASCAL VOC 2012)
            :param image_mask_transforms        transforms to be applied on image and mask when augment=False
            :param image_mask_transforms_aug    transforms to be applied on image and mask when augment=True
        """
        self.samples_sub_directory = samples_sub_directory
        self.targets_sub_directory = targets_sub_directory
        self.dataset_hyperparams = dataset_hyper_params
        self.cache_labels = cache_labels
        self.cache_images = cache_images
        self.batch_size = batch_size
        self.img_size = img_size
        self.crop_size = crop_size
        self.augment = augment
        self.batch_index = None
        self.total_batches_num = None

        # ENABLES USING CUSTOM SAMPLE/TARGET LOADERS
        if sample_loader is not None:
            self.sample_loader = sample_loader
        if target_loader is not None:
            self.target_loader = target_loader

        # CREATE A DIRECTORY DATASET OR A LIST DATASET BASED ON THE list_file INPUT VARIABLE
        if list_file is not None:
            ListDataset.__init__(self, root=root, file=list_file, target_extension=target_extension,
                                 sample_loader=self.sample_loader, sample_transform=self.sample_transform,
                                 target_loader=self.target_loader, target_transform=self.target_transform,
                                 collate_fn=collate_fn)
        else:
            DirectoryDataSet.__init__(self, root=root, samples_sub_directory=samples_sub_directory,
                                      targets_sub_directory=targets_sub_directory, target_extension=target_extension,
                                      sample_loader=self.sample_loader, sample_transform=self.sample_transform,
                                      target_loader=self.target_loader, target_transform=self.target_transform,
                                      collate_fn=collate_fn)
        # DEFAULT TRANSFORMS
        # FIXME - Rescale before RandomRescale is kept for legacy support, consider removing it like most implementation
        #  papers regimes.
        default_image_mask_transforms_aug = transform.Compose([RandomFlip(),
                                                               Rescale(short_size=self.img_size),
                                                               RandomRescale(scales=(0.5, 2.0)),
                                                               RandomRotate(),
                                                               PadShortToCropSize(self.crop_size),
                                                               CropImageAndMask(crop_size=self.crop_size,
                                                                                mode="random"),
                                                               RandomGaussianBlur()])

        self.image_mask_transforms_aug = image_mask_transforms_aug or default_image_mask_transforms_aug

        default_image_mask_transforms = transform.Compose([Rescale(short_size=self.crop_size),
                                                           CropImageAndMask(crop_size=self.crop_size, mode="center")
                                                           ])

        self.image_mask_transforms = image_mask_transforms or default_image_mask_transforms

    def __getitem__(self, index):
        sample_path, target_path = self.samples_targets_tuples_list[index]

        # TRY TO LOAD THE CACHED IMAGE FIRST
        if self.cache_images:
            sample = self.imgs[index]
        else:
            sample = self.sample_loader(sample_path)

        # TRY TO LOAD THE CACHED LABEL FIRST
        if self.cache_labels:
            target = self.labels[index]
        else:
            target = self.target_loader(target_path)

        # MAKE SURE THE TRANSFORM WORKS ON BOTH IMAGE AND MASK TO ALIGN THE AUGMENTATIONS
        sample, target = self._transform_image_and_mask(sample, target)

        return self.sample_transform(sample), self.target_transform(target)

    @staticmethod
    def sample_loader(sample_path: str) -> Image:
        """
        sample_loader - Loads a dataset image from path using PIL
            :param sample_path: The path to the sample image
            :return:            The loaded Image
        """
        image = Image.open(sample_path).convert('RGB')
        return image

    @staticmethod
    def sample_transform(image):
        """
        sample_transform - Transforms the sample image

            :param image:  The input image to transform
            :return:       The transformed image
        """
        sample_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])

        return sample_transform(image)

    @staticmethod
    def target_loader(target_path: str) -> Image:
        """
        target_loader
            :param target_path: The path to the sample image
            :return:            The loaded Image
        """
        target = Image.open(target_path)
        return target

    @staticmethod
    def target_transform(target):
        """
        target_transform - Transforms the sample image

            :param target: The target mask to transform
            :return:       The transformed target mask
        """
        return torch.from_numpy(np.array(target)).long()

    def _generate_samples_and_targets(self):
        """
        _generate_samples_and_targets
        """
        # IF THE DERIVED CLASS DID NOT IMPLEMENT AN EXPLICIT _generate_samples_and_targets CHILD METHOD
        if not self.samples_targets_tuples_list:
            super()._generate_samples_and_targets()

        self.batch_index = np.floor(np.arange(len(self)) / self.batch_size).astype(np.int)
        self.total_batches_num = self.batch_index[-1] + 1

        # EXTRACT THE LABELS FROM THE TUPLES LIST
        image_files, label_files = map(list, zip(*self.samples_targets_tuples_list))
        image_indices_to_remove = []

        # CACHE IMAGES INTO MEMORY FOR FASTER TRAINING (WARNING: LARGE DATASETS MAY EXCEED SYSTEM RAM)
        if self.cache_images:
            # CREATE AN EMPTY LIST FOR THE LABELS
            self.imgs = len(self) * [None]
            cached_images_mem_in_gb = 0.
            pbar = tqdm(image_files, desc='Caching images')
            for i, img_path in enumerate(pbar):
                img = self.sample_loader(img_path)
                if img is None:
                    image_indices_to_remove.append(i)

                cached_images_mem_in_gb += os.path.getsize(image_files[i]) / 1024. ** 3.

                self.imgs[i] = img
                pbar.desc = 'Caching images (%.1fGB)' % (cached_images_mem_in_gb)
            self.img_files = [e for i, e in enumerate(image_files) if i not in image_indices_to_remove]
            self.imgs = [e for i, e in enumerate(self.imgs) if i not in image_indices_to_remove]

        # CACHE LABELS INTO MEMORY FOR FASTER TRAINING - RELEVANT FOR EFFICIENT VALIDATION RUNS DURING TRAINING
        if self.cache_labels:
            # CREATE AN EMPTY LIST FOR THE LABELS
            self.labels = len(self) * [None]
            pbar = tqdm(label_files, desc='Caching labels')
            missing_labels, found_labels, duplicate_labels = 0, 0, 0

            for i, file in enumerate(pbar):
                labels = self.target_loader(file)

                if labels is None:
                    missing_labels += 1
                    image_indices_to_remove.append(i)
                    continue

                self.labels[i] = labels
                found_labels += 1

                pbar.desc = 'Caching labels (%g found, %g missing, %g duplicate, for %g images)' % (
                    found_labels, missing_labels, duplicate_labels, len(image_files))
            assert found_labels > 0, 'No labels found.'

            #  REMOVE THE IRRELEVANT ENTRIES FROM THE DATA
            self.label_files = [e for i, e in enumerate(label_files) if i not in image_indices_to_remove]
            self.labels = [e for i, e in enumerate(self.labels) if i not in image_indices_to_remove]

    def _calculate_short_size(self, img):
        """
        _calculate_crop
        :param img:
        :return:
        """
        if self.augment:
            # RANDOM SCALE (SHORT EDGE FROM 480 TO 720)
            short_size = random.randint(int(self.img_size * 0.5), int(self.img_size * 2.0))
        else:
            short_size = self.crop_size

        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)

        return oh, ow, short_size

    def _get_center_crop(self, w, h):
        """

        :param w:
        :param h:
        :return:
        """
        # CENTER CROP
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))

        if self.augment:
            # RANDOM CROP CROP_SIZE
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)

        return x1, y1

    def _transform_image_and_mask(self, image, mask) -> tuple:
        """
        _transform -  Transforms the input (image, mask) in the following order:
                                1. FLIP (if augment==true)
                                2. RESIZE
                                3. ROTATE (if augment==true)
                                4. CROP
                                5. GAUSSIAN BLUR (if augment==true)

                            * Please use self.augment == True only for training

            :param image:           The input image
            :param mask:            The input mask
            :return:                The transformed image, mask
        """
        if self.augment:
            transformed = self.image_mask_transforms_aug({"image": image, "mask": mask})
        else:
            transformed = self.image_mask_transforms({"image": image, "mask": mask})

        return transformed["image"], transformed["mask"]
