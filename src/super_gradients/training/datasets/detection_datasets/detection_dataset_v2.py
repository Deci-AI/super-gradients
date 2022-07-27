import os
from typing import List, Dict, Union, Any, Optional, Tuple
import random
import cv2
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from super_gradients.training.utils.detection_utils import get_label_posx_in_target, DetectionTargetsFormat
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.transforms.transforms import DetectionTransform

logger = get_logger(__name__)


class DetectionDataSetV2(Dataset):
    """Detection dataset.

    This is a boilerplate class with a premade workflow to facilitate the implementation of datasets.
    The workflow is as follow:
    - On instatiation:
        - All the (raw) annotations are loaded in the ram. If class_inclusion_list, there is also subclassing at this step.
        - If cache is True, the images as well

    - On call (__getitem__) for a specific image:
        - The image and annotations are fetched

    INDEX VS SAMPLE_ID
    - index refers to the index in the dataset.
    - sample_id refers to the id of sample before droping any annotaion
    Let's imagine a situation where the downloaded data is made of 120 annotations but 20 were droped because
    some images had no annotation. In that case:
        > len(self) = 100
        > index will be between 0 and 100
        > sample_id will be between 0 to 120
    """

    def __init__(
            self,
            input_dim: tuple,
            n_available_samples: int,
            target_format: DetectionTargetsFormat,
            cache: bool = False,
            cache_path: str = None,
            transforms: List[DetectionTransform] = [],
            class_inclusion_list: Optional[List[str]] = None,
            keep_empty_annotations: bool = False,
            all_classes_list: Optional[List[str]] = None,
            annotation_fields_to_subclass: List[str] = None,
            output_fields: List[str] = None,
    ):
        """
        DetectionDataSetV2

        :param input_dim:            Image size (when loaded, before transforms)
        :param cache:                Whether to cache images
        :param cache_path:       Path to a directory that will be used for caching (with memmap)
        :param transforms:           List of transforms to apply sequentially on sample in __getitem__
        :param all_classes_list:     All the class names.
        :param class_inclusion_list: Subclass names or None when subclassing is disabled.
        """
        super().__init__()

        self.input_dim = input_dim
        self.target_format = target_format
        self.transforms = transforms

        # n_available_samples can be different to len(self.annotations) and len(self) if annotations are drop.
        self.n_available_samples = n_available_samples

        self.all_classes_list = all_classes_list
        self.class_inclusion_list = class_inclusion_list
        self.classes = self.class_inclusion_list or self.all_classes_list

        self.keep_empty_annotations = keep_empty_annotations
        self.annotation_fields_to_subclass = annotation_fields_to_subclass or ["target"]
        assert "target" in self.annotation_fields_to_subclass,\
            '"target" was not included in annotation_fields_to_subclass'

        self.annotations = self._cache_annotations()
        assert len(self.annotations) > 0

        self.cache = cache
        self.cache_path = cache_path
        self.cached_imgs = self._cache_images() if self.cache else None

        self.output_fields = output_fields or ["image", "target"]
        assert "image" in self.output_fields, '"image" is expected to be in output_fields but it was not included'
        assert "target" in self.output_fields, '"target" is expected to be in output_fields but it was not included'

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        """
        Loads annotation associated to one sample (i.e. couple image/ground_truth)
        :param sample_id:   sample_id
        :return:            Annotation, a dict with any field but has to include at least "target" and "img_path".
        """
        raise NotImplementedError

    def _cache_annotations(self) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Load all the annotations to memory to avoid opening files back and forth.
        This step is done in the dataset initalization.

        :return:
        """
        annotations = []
        for sample_id, img_id in enumerate(tqdm(range(self.n_available_samples), desc="Caching annotations")):
            img_annotation = self._load_annotation(img_id)
            assert "target" in img_annotation, '_load_annotation  is expected to return the field "target"'
            assert "img_path" in img_annotation, '_load_annotation is expected to return the field "img_path"'

            if self.class_inclusion_list is not None:
                img_annotation = self._sub_class_annotation(img_annotation)
            if self.keep_empty_annotations or img_annotation is not None:
                annotations.append(img_annotation)
        return annotations

    def _sub_class_target(self, targets: np.ndarray, label_posx) -> np.ndarray:
        """Sublass targets of a specific image."""

        targets_kept = []
        for target in targets:
            label_id = int(target[label_posx])
            label_name = self.all_classes_list[label_id]
            if label_name in self.class_inclusion_list:
                # Replace the target label_id in self.all_classes_list by label_id in self.class_inclusion_list
                target[label_posx] = self.class_inclusion_list.index(label_name)
                targets_kept.append(target)

        return np.array(targets_kept) if len(targets_kept) > 0 else np.zeros((0, 5), dtype=np.float32)

    def _sub_class_annotation(self, annotation: dict) -> Union[dict, None]:
        """
        Subclass every field listed in self.annotation_fields_to_subclass.
        It could be targets, crowd_targets, ect ...
        """
        label_posx = get_label_posx_in_target(self.target_format)
        for field in self.annotation_fields_to_subclass:
            annotation[field] = self._sub_class_target(targets=annotation[field], label_posx=label_posx)

        is_annotation_non_empty = any(len(annotation[field]) > 0 for field in self.annotation_fields_to_subclass)
        return annotation if (self.keep_empty_annotations or is_annotation_non_empty) else None

    def _load_image(self, index: int) -> np.ndarray:
        """
        Loads image at index with its original resolution.
        :param index: index in self.annotations
        :return: image (np.ndarray)
        """
        img_path = self.annotations[index]["img_path"]

        img_file = os.path.join(img_path)
        img = cv2.imread(img_file)

        assert img is not None, \
            f"{img_file} was no found. Please make sure that the dataset was downloaded and that the path is correct"
        return img

    def _load_resized_img(self, index: int) -> np.ndarray:
        """
        Loads image at index, and resizes it to self.input_dim

        :param index: sample_id to load the image from
        :return: resized_img
        """
        img = self._load_image(index)

        r = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
        desired_size = (int(img.shape[1] * r), int(img.shape[0] * r))

        resized_img = cv2.resize(src=img, dsize=desired_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        return resized_img

    def _cache_images(self) -> np.ndarray:
        cache_path = Path(self.cache_path)
        if cache_path is None or not cache_path.parent.exists():
            raise ValueError("Must pass valid path through cache_path when caching. Got " + str(cache_path))
        if cache_path.parent.exists() and not cache_path.exists():
            cache_path.mkdir()

        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h, max_w = self.input_dim[0], self.input_dim[1]
        cache_file = cache_path / f"img_resized_cache.array"

        if not cache_file.exists():
            logger.info("Caching images for the first time.")
            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(func=lambda x: self._load_resized_img(x),
                                                         iterable=range(len(self)))

            # Initialize placeholder for images
            cached_imgs = np.memmap(cache_file, shape=(len(self), max_h, max_w, 3),
                                    dtype=np.uint8, mode="w+")

            # Store images in the placeholder
            loaded_images_pbar = tqdm(enumerate(loaded_images), total=len(self))
            for i, image in loaded_images_pbar:
                cached_imgs[i][: image.shape[0], : image.shape[1], :] = image.copy()
            cached_imgs.flush()
            loaded_images_pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        cached_imgs = np.memmap(cache_file, shape=(len(self), max_h, max_w, 3),
                                dtype=np.uint8, mode="r+")
        return cached_imgs

    def __del__(self):
        """Clear the cached images"""
        if hasattr(self, "cached_imgs"):
            del self.cached_imgs

    def __len__(self):
        """
        Get the length of the dataset.
        This can be modified in _load_annotation when keep_empty_annotations=False and class_inclusion_list is not None.
        """
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple:
        """Method called by SgModel."""
        sample = self.apply_transforms(self.get_sample(index))
        for field in self.output_fields:
            assert field in sample, f'The field {field} must be present in the sample but was not found.'\
                                     'Please check the output fields of your transforms.'
        return tuple(sample[field] for field in self.output_fields)

    def get_sample(self, index: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Get raw sample, before transforms, in a dict format."""
        img = self.get_resized_image(index)
        annotation = self.annotations[index]
        return {"image": img, **annotation}

    def get_resized_image(self, index: int) -> np.ndarray:  # TODO: Check if this the standard way, because COCO is different
        """
        Get the resized image at a specific sample_id, either from cache or by loading from disk, based on self.cached_imgs
        :param index:  Image index
        :return:       Resized image
        """
        if self.cached_imgs is not None:
            return self.cached_imgs[index].copy()
        else:
            return self._load_resized_img(index)

    def apply_transforms(self, sample: Dict[str, Union[np.ndarray, Any]]) -> Dict[str, Union[np.ndarray, Any]]:
        """
        Applies self.transforms sequentially to sample

        If a transforms has the attribute 'additional_samples_count', additional samples will be loaded and stored in
         sample["additional_samples"] prior to applying it. Combining with the attribute "non_empty_annotations" will load
         only additional samples with objects in them.

        :param sample: Sample to apply the transforms on to (loaded with self.get_sample)
        :return: Transformed sample
        """
        for transform in self.transforms:
            self._add_additional_inputs_for_transform(sample, transform)
            sample = transform(sample)
        sample.pop("additional_samples")  # additional_samples is not useful after the transforms
        return sample

    def _add_additional_inputs_for_transform(
            self, sample: Dict[str, Union[np.ndarray, Any]], transform: DetectionTransform):
        additional_samples_count = transform.additional_samples_count if hasattr(transform,
                                                                                 "additional_samples_count") else 0
        non_empty_annotations = transform.non_empty_annotations if hasattr(transform, "non_empty_annotations") else False
        additional_samples = self._get_random_samples(additional_samples_count, non_empty_annotations)
        sample["additional_samples"] = additional_samples

    def _get_random_samples(
            self, count: int, non_empty_annotations_only: bool = False) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Load random samples

        :param count: The number of samples wanted
        :param non_empty_annotations_only: If true, only return samples with at least 1 annotation
        :return: A list of samples satisfying input params
        """
        indexes = [
            self._get_random_non_empty_annotation_available_indexes() if non_empty_annotations_only else self._get_random_index()
            for _ in range(count)]
        return [self.get_sample(index) for index in indexes]

    def _get_random_non_empty_annotation_available_indexes(self) -> int:
        """Get the sample_id of a non-empty annotation"""
        target, index = [], -1
        while len(target) == 0:
            index = self._get_random_index()
            target = self.annotations[index]["target"]
        return index

    def _get_random_index(self) -> int:
        return random.randint(0, len(self) - 1)
