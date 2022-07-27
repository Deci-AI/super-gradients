import os
from typing import List, Dict, Union, Any, Optional, Tuple
import random
import cv2
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from super_gradients.training.utils.detection_utils import get_cls_posx_in_target, DetectionTargetsFormat
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.transforms.transforms import DetectionTransform

logger = get_logger(__name__)


class DetectionDataSetV2(Dataset):
    """Detection dataset V2.

    This is a boilerplate class with a premade workflow to facilitate the implementation of datasets.

    HOW TO CREATE A DATASET THAT INHERITS FROM DetectionDataSetV2 ?
        - Inherit from DetectionDataSetV2
        - implement the method self._load_annotation to return at least the fields "target" and "img_path"
        - Call super().__init__ with the reauired params at the end of.
                //!\\ super().__init__ will call self._load_annotation, so make sure that every required
                      attributes are set up before calling super().__init__ (ideally just call it last)

    WORKFLOW:
        - On instantiation:
            - All annotations are cached. If class_inclusion_list was specified, there is also subclassing at this step.
            - If cache is True, the images are also cached

        - On call (__getitem__) for a specific image index:
            - The image and annotations are grouped together in a dict called SAMPLE
            - the sample is processed according to th transform
            - Only the specified fields are returned by __getitem__

    TERMINOLOGY
        - TARGET:       Groundtruth, made of bboxes. The format can vary from one dataset to another
        - ANNOTATION:   Combination of targets (groundtruth) and metadata of the image, but without the image itself.
                            > Has to include the fields "target" and "img_path"
                            > Can include other fields like "crowd_target", "image_info", "segmentation", ...
        - SAMPLE:       Outout of the dataset:
                            > Has to include the fields "target" and "image"
                            > Can include other fields like "crowd_target", "image_info", "segmentation", ...
        - INDEX:        Refers to the index in the dataset.
        - SAMPLE ID:    Refers to the id of sample before droping any annotaion.
                            Let's imagine a situation where the downloaded data is made of 120 images but 20 were drop
                            because they had no annotation. In that case:
                                > We have 120 samples so sample_id will be between 0 and 119
                                > But only 100 will be indexed so index will be between 0 and 99
                                > Therefore, we also have len(self) = 100
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
            '"target" is expected to be in the fields to subclassbut it was not included'

        self.annotations = self._cache_annotations()
        assert len(self.annotations) > 0

        self.cache = cache
        self.cache_path = cache_path
        self.cached_imgs = self._cache_images() if self.cache else None

        self.output_fields = output_fields or ["image", "target"]
        assert "image" in self.output_fields, '"image" is expected to be in output_fields but it was not included'
        assert "target" in self.output_fields, '"target" is expected to be in output_fields but it was not included'

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Load annotations associated to a specific sample.

        :param sample_id:   Id of the sample to load annotations from.
        :return:            Annotation, a dict with any field but has to include at least "target" and "img_path".
        """
        raise NotImplementedError

    def _cache_annotations(self) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Load all the annotations to memory to avoid opening files back and forth.
        :return: List of annotations
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

    def _sub_class_annotation(self, annotation: dict) -> Union[dict, None]:
        """Subclass every field listed in self.annotation_fields_to_subclass. It could be targets, crowd_targets, ...

        :param annotation: Dict representing the annotation of a specific image
        :return:           Subclassed annotation if non empty after subclassing, otherwise None
        """
        cls_posx = get_cls_posx_in_target(self.target_format)
        for field in self.annotation_fields_to_subclass:
            annotation[field] = self._sub_class_target(targets=annotation[field], cls_posx=cls_posx)

        is_annotation_non_empty = any(len(annotation[field]) > 0 for field in self.annotation_fields_to_subclass)
        return annotation if (self.keep_empty_annotations or is_annotation_non_empty) else None

    def _sub_class_target(self, targets: np.ndarray, cls_posx: int) -> np.ndarray:
        """Sublass targets of a specific image.

        :param targets:     Target array to subclass of shape [n_targets, 5], 5 representing a bbox
        :param cls_posx:    Position of the class id in a bbox
                                ex: 0 if bbox of format label_xyxy | -1 if bbox of format xyxy_label
        :return:            Subclassed target
        """
        targets_kept = []
        for target in targets:
            cls_id = int(target[cls_posx])
            cls_name = self.all_classes_list[cls_id]
            if cls_name in self.class_inclusion_list:
                # Replace the target cls_id in self.all_classes_list by cls_id in self.class_inclusion_list
                target[cls_posx] = self.class_inclusion_list.index(cls_name)
                targets_kept.append(target)

        return np.array(targets_kept) if len(targets_kept) > 0 else np.zeros((0, 5), dtype=np.float32)

    def _cache_images(self) -> np.ndarray:
        """Cache the images. The cached image are stored in a file to be loaded faster mext time.
        :return: Cached images
        """
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
        img_resized_cache_path = cache_path / f"img_resized_cache.array"

        if not img_resized_cache_path.exists():
            logger.info("Caching images for the first time.")
            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(func=lambda x: self._load_resized_img(x),
                                                         iterable=range(len(self)))

            # Initialize placeholder for images
            cached_imgs = np.memmap(str(img_resized_cache_path), shape=(len(self), max_h, max_w, 3),
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
        cached_imgs = np.memmap(str(img_resized_cache_path), shape=(len(self), max_h, max_w, 3),
                                dtype=np.uint8, mode="r+")
        return cached_imgs

    def _load_resized_img(self, index: int) -> np.ndarray:
        """Load image, and resizes it to self.input_dim
        :param index:   Image index
        :return:        Resized image
        """
        img = self._load_image(index)

        r = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
        desired_size = (int(img.shape[1] * r), int(img.shape[0] * r))

        resized_img = cv2.resize(src=img, dsize=desired_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        return resized_img

    def _load_image(self, index: int) -> np.ndarray:
        """Loads image at index with its original resolution.
        :param index:   Image index
        :return:        Image in array format
        """
        img_path = self.annotations[index]["img_path"]

        img_file = os.path.join(img_path)
        img = cv2.imread(img_file)

        assert img is not None, \
            f"{img_file} was no found. Please make sure that the dataset was downloaded and that the path is correct"
        return img

    def __del__(self):
        """Clear the cached images"""
        if hasattr(self, "cached_imgs"):
            del self.cached_imgs

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple:
        """Get the sample at a specific index of the dataset.
        The output of this function will be collated to form batches."""
        sample = self.apply_transforms(self.get_sample(index))
        for field in self.output_fields:
            assert field in sample, f'The field {field} must be present in the sample but was not found.'\
                                     'Please check the output fields of your transforms.'
        return tuple(sample[field] for field in self.output_fields)

    def get_sample(self, index: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Get raw sample, before any transform (beside subclassing).
        :param index:   Image index
        :return:        Sample, i.e. a dictionary including at least "image" and "target"
        """
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

    def _add_additional_inputs_for_transform(self, sample: Dict[str, Union[np.ndarray, Any]],
                                             transform: DetectionTransform):
        """Add additional inputs required by a transform to the sample"""
        additional_samples_count = transform.additional_samples_count if hasattr(transform,
                                                                                 "additional_samples_count") else 0
        non_empty_annotations = transform.non_empty_annotations if hasattr(transform, "non_empty_annotations") else False
        additional_samples = self._get_random_samples(additional_samples_count, non_empty_annotations)
        sample["additional_samples"] = additional_samples

    def _get_random_samples(
            self, count: int, non_empty_annotations_only: bool = False) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Load random samples.

        :param count: The number of samples wanted
        :param non_empty_annotations_only: If true, only return samples with at least 1 annotation
        :return: A list of samples satisfying input params
        """
        indexes = [
            self._get_random_non_empty_annotation_available_indexes() if non_empty_annotations_only else self._get_random_index()
            for _ in range(count)]
        return [self.get_sample(index) for index in indexes]

    def _get_random_non_empty_annotation_available_indexes(self) -> int:
        """Get the index of a non-empty annotation.
        :return: Image index"""
        target, index = [], -1
        while len(target) == 0:
            index = self._get_random_index()
            target = self.annotations[index]["target"]
        return index

    def _get_random_index(self) -> int:
        """Get a random index of this dataset.
        :return: Random image index"""
        return random.randint(0, len(self) - 1)
