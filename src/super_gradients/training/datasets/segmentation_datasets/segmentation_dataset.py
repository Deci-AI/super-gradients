import os
from typing import Callable, Iterable

import numpy as np
import torch
import torchvision.transforms as transform
from PIL import Image
from tqdm import tqdm

from super_gradients.common.object_names import Datasets
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.datasets.sg_dataset import DirectoryDataSet, ListDataset


@register_dataset(Datasets.SEGMENTATION_DATASET)
class SegmentationDataSet(DirectoryDataSet, ListDataset):
    @resolve_param("transforms", factory=TransformsFactory())
    def __init__(
        self,
        root: str,
        list_file: str = None,
        samples_sub_directory: str = None,
        targets_sub_directory: str = None,
        cache_labels: bool = False,
        cache_images: bool = False,
        collate_fn: Callable = None,
        target_extension: str = ".png",
        transforms: Iterable = None,
    ):
        """
        SegmentationDataSet
            :param root:                        Root folder of the Data Set
            :param list_file:                   Path to the file with the samples list
            :param samples_sub_directory:       name of the samples sub-directory
            :param targets_sub_directory:       name of the targets sub-directory
            :param cache_labels:                "Caches" the labels -> Pre-Loads to memory as a list
            :param cache_images:                "Caches" the images -> Pre-Loads to memory as a list
            :param collate_fn:                  collate_fn func to process batches for the Data Loader
            :param target_extension:            file extension of the targets (default is .png for PASCAL VOC 2012)
            :param transforms:                  transforms to be applied on image and mask

        """
        self.samples_sub_directory = samples_sub_directory
        self.targets_sub_directory = targets_sub_directory
        self.cache_labels = cache_labels
        self.cache_images = cache_images

        # CREATE A DIRECTORY DATASET OR A LIST DATASET BASED ON THE list_file INPUT VARIABLE
        if list_file is not None:
            ListDataset.__init__(
                self,
                root=root,
                file=list_file,
                target_extension=target_extension,
                sample_loader=self.sample_loader,
                sample_transform=self.sample_transform,
                target_loader=self.target_loader,
                target_transform=self.target_transform,
                collate_fn=collate_fn,
            )
        else:
            DirectoryDataSet.__init__(
                self,
                root=root,
                samples_sub_directory=samples_sub_directory,
                targets_sub_directory=targets_sub_directory,
                target_extension=target_extension,
                sample_loader=self.sample_loader,
                sample_transform=self.sample_transform,
                target_loader=self.target_loader,
                target_transform=self.target_transform,
                collate_fn=collate_fn,
            )

        self.transforms = transform.Compose(transforms if transforms else [])

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
        image = Image.open(sample_path).convert("RGB")
        return image

    @staticmethod
    def sample_transform(image):
        """
        sample_transform - Transforms the sample image

            :param image:  The input image to transform
            :return:       The transformed image
        """
        sample_transform = transform.Compose([transform.ToTensor(), transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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

        # EXTRACT THE LABELS FROM THE TUPLES LIST
        image_files, label_files = map(list, zip(*self.samples_targets_tuples_list))
        image_indices_to_remove = []

        # CACHE IMAGES INTO MEMORY FOR FASTER TRAINING (WARNING: LARGE DATASETS MAY EXCEED SYSTEM RAM)
        if self.cache_images:
            # CREATE AN EMPTY LIST FOR THE LABELS
            self.imgs = len(self) * [None]
            cached_images_mem_in_gb = 0.0
            with tqdm(image_files, desc="Caching images") as pbar:
                for i, img_path in enumerate(pbar):
                    img = self.sample_loader(img_path)
                    if img is None:
                        image_indices_to_remove.append(i)

                    cached_images_mem_in_gb += os.path.getsize(image_files[i]) / 1024.0**3.0

                    self.imgs[i] = img
                    pbar.desc = "Caching images (%.1fGB)" % (cached_images_mem_in_gb)
            self.img_files = [e for i, e in enumerate(image_files) if i not in image_indices_to_remove]
            self.imgs = [e for i, e in enumerate(self.imgs) if i not in image_indices_to_remove]

        # CACHE LABELS INTO MEMORY FOR FASTER TRAINING - RELEVANT FOR EFFICIENT VALIDATION RUNS DURING TRAINING
        if self.cache_labels:
            # CREATE AN EMPTY LIST FOR THE LABELS
            self.labels = len(self) * [None]
            with tqdm(label_files, desc="Caching labels") as pbar:
                missing_labels, found_labels, duplicate_labels = 0, 0, 0

                for i, file in enumerate(pbar):
                    labels = self.target_loader(file)

                    if labels is None:
                        missing_labels += 1
                        image_indices_to_remove.append(i)
                        continue

                    self.labels[i] = labels
                    found_labels += 1

                    pbar.desc = "Caching labels (%g found, %g missing, %g duplicate, for %g images)" % (
                        found_labels,
                        missing_labels,
                        duplicate_labels,
                        len(image_files),
                    )
            assert found_labels > 0, "No labels found."

            #  REMOVE THE IRRELEVANT ENTRIES FROM THE DATA
            self.label_files = [e for i, e in enumerate(label_files) if i not in image_indices_to_remove]
            self.labels = [e for i, e in enumerate(self.labels) if i not in image_indices_to_remove]

    def _transform_image_and_mask(self, image, mask) -> tuple:
        """
        :param image:           The input image
        :param mask:            The input mask
        :return:                The transformed image, mask
        """
        transformed = self.transforms({"image": image, "mask": mask})
        return transformed["image"], transformed["mask"]
