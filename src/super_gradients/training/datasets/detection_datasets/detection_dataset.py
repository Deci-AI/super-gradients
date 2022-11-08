import os
from typing import List, Dict, Union, Any, Optional, Tuple
from multiprocessing.pool import ThreadPool
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import hashlib

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.training.utils.detection_utils import get_cls_posx_in_target, DetectionTargetsFormat
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.transforms.transforms import DetectionTransform, DetectionTargetsFormatTransform
from super_gradients.training.exceptions.dataset_exceptions import EmptyDatasetException
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory

logger = get_logger(__name__)


class DetectionDataset(Dataset):
    """Detection dataset.

    This is a boilerplate class to facilitate the implementation of datasets.

    HOW TO CREATE A DATASET THAT INHERITS FROM DetectionDataSet ?
        - Inherit from DetectionDataSet
        - implement the method self._load_annotation to return at least the fields "target" and "img_path"
        - Call super().__init__ with the required params.
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

    @resolve_param("transforms", ListFactory(TransformsFactory()))
    def __init__(
        self,
        data_dir: str,
        input_dim: tuple,
        original_target_format: DetectionTargetsFormat,
        max_num_samples: int = None,
        cache: bool = False,
        cache_dir: str = None,
        transforms: List[DetectionTransform] = [],
        all_classes_list: Optional[List[str]] = None,
        class_inclusion_list: Optional[List[str]] = None,
        ignore_empty_annotations: bool = True,
        target_fields: List[str] = None,
        output_fields: List[str] = None,
    ):
        """Detection dataset.

        :param data_dir:                Where the data is stored
        :param input_dim:               Image size (when loaded, before transforms).
        :param original_target_format:  Format of targets stored on disk. raw data format, the output format might
                                        differ based on transforms.
        :param max_num_samples:         If not None, set the maximum size of the dataset by only indexing the first n annotations/images.
        :param cache:                   Whether to cache images or not.
        :param cache_dir:              Path to the directory where cached images will be stored in an optimized format.
        :param transforms:              List of transforms to apply sequentially on sample.
        :param all_classes_list:        All the class names.
        :param class_inclusion_list:    If not None,every class not included will be ignored.
        :param ignore_empty_annotations:        If True and class_inclusion_list not None, images without any target
                                                will be ignored.
        :param target_fields:                   List of the fields target fields. This has to include regular target,
                                                but can also include crowd target, segmentation target, ...
                                                It has to include at least "target" but can include other.
        :paran output_fields:                   Fields that will be outputed by __getitem__.
                                                It has to include at least "image" and "target" but can include other.
        """
        super().__init__()

        self.data_dir = data_dir
        if not Path(data_dir).exists():
            raise FileNotFoundError(f"Please make sure to download the data in the data directory ({self.data_dir}).")

        # Number of images that are avalaible(regardless of ignored images)
        self.n_available_samples = self._setup_data_source()
        if not isinstance(self.n_available_samples, int) or self.n_available_samples < 1:
            raise ValueError(f"_setup_data_source() should return the number of available samples but got {self.n_available_samples}")

        self.input_dim = input_dim
        self.original_target_format = original_target_format
        self.max_num_samples = max_num_samples

        self.all_classes_list = all_classes_list
        self.class_inclusion_list = class_inclusion_list
        self.classes = self.class_inclusion_list or self.all_classes_list
        if len(set(self.classes) - set(all_classes_list)) > 0:
            wrong_classes = set(self.classes) - set(all_classes_list)
            raise ValueError(f"class_inclusion_list includes classes that are not in all_classes_list: {wrong_classes}")

        self.ignore_empty_annotations = ignore_empty_annotations
        self.target_fields = target_fields or ["target"]
        if "target" not in self.target_fields:
            raise KeyError('"target" is expected to be in the fields to subclass but it was not included')

        self._required_annotation_fields = {"target", "img_path", "resized_img_shape"}
        self.annotations = self._cache_annotations()

        self.cache = cache
        self.cache_dir = cache_dir
        self.cached_imgs_padded = self._cache_images() if self.cache else None

        self.transforms = transforms

        self.output_fields = output_fields or ["image", "target"]
        if len(self.output_fields) < 2 or self.output_fields[0] != "image" or self.output_fields[1] != "target":
            raise ValueError('output_fields must start with "image" and then "target", followed by any other field')

    def _setup_data_source(self) -> int:
        """Set up the data source and store relevant objects as attributes.

        :return: Number of available samples, (i.e. how many images we have, regardless of any filter we might want to use)"""
        raise NotImplementedError

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Load annotations associated to a specific sample.
        Please note that the targets should be resized according to self.input_dim!

        :param sample_id:   Id of the sample to load annotations from.
        :return:            Annotation, a dict with any field but has to include at least the fields specified in self._required_annotation_fields.
        """
        raise NotImplementedError

    def _cache_annotations(self) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Load all the annotations to memory to avoid opening files back and forth.
        :return: List of annotations
        """
        annotations = []
        for sample_id, img_id in enumerate(tqdm(range(self.n_available_samples), desc="Caching annotations")):

            if self.max_num_samples is not None and len(annotations) >= self.max_num_samples:
                break

            img_annotation = self._load_annotation(img_id)
            if not self._required_annotation_fields.issubset(set(img_annotation.keys())):
                raise KeyError(
                    f"_load_annotation is expected to return at least the fields {self._required_annotation_fields} " f"but got {set(img_annotation.keys())}"
                )

            if self.class_inclusion_list is not None:
                img_annotation = self._sub_class_annotation(img_annotation)

            is_annotation_empty = all(len(img_annotation[field]) == 0 for field in self.target_fields)
            if self.ignore_empty_annotations and is_annotation_empty:
                continue
            annotations.append(img_annotation)

        if len(annotations) == 0:
            raise EmptyDatasetException(
                f"Out of {self.n_available_samples} images, not a single one was found with" f"any of these classes: {self.class_inclusion_list}"
            )
        return annotations

    def _sub_class_annotation(self, annotation: dict) -> Union[dict, None]:
        """Subclass every field listed in self.target_fields. It could be targets, crowd_targets, ...

        :param annotation: Dict representing the annotation of a specific image
        :return:           Subclassed annotation if non empty after subclassing, otherwise None
        """
        cls_posx = get_cls_posx_in_target(self.original_target_format)
        for field in self.target_fields:
            annotation[field] = self._sub_class_target(targets=annotation[field], cls_posx=cls_posx)
        return annotation

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
        cache_dir = Path(self.cache_dir)
        if cache_dir is None:
            raise ValueError("You must specify a cache_dir if you want to cache your images." "If you did not mean to use cache, please set cache=False ")
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "********************************************************************************"
        )

        max_h, max_w = self.input_dim[0], self.input_dim[1]

        # The cache should be the same as long as the images and their sizes are the same
        hash = hashlib.sha256()
        for annotation in self.annotations:
            values_to_hash = [annotation["resized_img_shape"][0], annotation["resized_img_shape"][1], Path(annotation["img_path"]).name]
            for value in values_to_hash:
                hash.update(str(value).encode("utf-8"))
        cache_hash = hash.hexdigest()

        img_resized_cache_path = cache_dir / f"img_resized_cache_{cache_hash}.array"

        if not img_resized_cache_path.exists():
            logger.info("Caching images for the first time. Be aware that this will stay in the disk until you delete it yourself.")
            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(func=lambda x: self._load_resized_img(x), iterable=range(len(self)))

            # Initialize placeholder for images
            cached_imgs = np.memmap(str(img_resized_cache_path), shape=(len(self), max_h, max_w, 3), dtype=np.uint8, mode="w+")

            # Store images in the placeholder
            loaded_images_pbar = tqdm(enumerate(loaded_images), total=len(self))
            for i, image in loaded_images_pbar:
                cached_imgs[i][: image.shape[0], : image.shape[1], :] = image.copy()
            cached_imgs.flush()
            loaded_images_pbar.close()
        else:
            logger.warning("You are using cached imgs!")

        logger.info("Loading cached imgs...")
        cached_imgs = np.memmap(str(img_resized_cache_path), shape=(len(self), max_h, max_w, 3), dtype=np.uint8, mode="r+")
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

        if img is None:
            raise FileNotFoundError(f"{img_file} was no found. Please make sure that the dataset was" f"downloaded and that the path is correct")
        return img

    def __del__(self):
        """Clear the cached images"""
        if hasattr(self, "cached_imgs_padded"):
            del self.cached_imgs_padded

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple:
        """Get the sample post transforms at a specific index of the dataset.
        The output of this function will be collated to form batches."""
        sample = self.apply_transforms(self.get_sample(index))
        for field in self.output_fields:
            if field not in sample.keys():
                raise KeyError(f"The field {field} must be present in the sample but was not found." "Please check the output fields of your transforms.")
        return tuple(sample[field] for field in self.output_fields)

    def get_random_item(self):
        return self[self._random_index()]

    def get_sample(self, index: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Get raw sample, before any transform (beside subclassing).
        :param index:   Image index
        :return:        Sample, i.e. a dictionary including at least "image" and "target"
        """
        img = self.get_resized_image(index)
        annotation = deepcopy(self.annotations[index])
        return {"image": img, **annotation}

    def get_resized_image(self, index: int) -> np.ndarray:
        """
        Get the resized image (i.e. either width or height reaches its input_dim) at a specific sample_id,
        either from cache or by loading from disk, based on self.cached_imgs_padded
        :param index:  Image index
        :return:       Resized image
        """
        if self.cache:
            padded_image = self.cached_imgs_padded[index]
            resized_height, resized_width = self.annotations[index]["resized_img_shape"]
            resized_image = padded_image[:resized_height, :resized_width, :]
            return resized_image.copy()
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
            sample.pop("additional_samples")  # additional_samples is not useful after the transform
        return sample

    def _add_additional_inputs_for_transform(self, sample: Dict[str, Union[np.ndarray, Any]], transform: DetectionTransform):
        """Add additional inputs required by a transform to the sample"""
        additional_samples_count = transform.additional_samples_count if hasattr(transform, "additional_samples_count") else 0
        non_empty_annotations = transform.non_empty_annotations if hasattr(transform, "non_empty_annotations") else False
        additional_samples = self.get_random_samples(additional_samples_count, non_empty_annotations)
        sample["additional_samples"] = additional_samples

    def get_random_samples(self, count: int, non_empty_annotations_only: bool = False) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Load random samples.

        :param count: The number of samples wanted
        :param non_empty_annotations_only: If true, only return samples with at least 1 annotation
        :return: A list of samples satisfying input params
        """
        return [self.get_random_sample(non_empty_annotations_only) for _ in range(count)]

    def get_random_sample(self, non_empty_annotations_only: bool = False):
        if non_empty_annotations_only:
            return self.get_sample(self._get_random_non_empty_annotation_available_indexes())
        else:
            return self.get_sample(self._random_index())

    def _get_random_non_empty_annotation_available_indexes(self) -> int:
        """Get the index of a non-empty annotation.
        :return: Image index"""
        target, index = [], -1
        while len(target) == 0:
            index = self._random_index()
            target = self.annotations[index]["target"]
        return index

    def _random_index(self):
        """Get a random index of this dataset"""
        return random.randint(0, len(self) - 1)

    @property
    def output_target_format(self):
        target_format = self.original_target_format
        for transform in self.transforms:
            if isinstance(transform, DetectionTargetsFormatTransform):
                target_format = transform.output_format
        return target_format

    def plot(self, max_samples_per_plot: int = 16, n_plots: int = 1, plot_transformed_data: bool = True):
        """Combine samples of images with bbox into plots and display the result.

        :param max_samples_per_plot:    Maximum number of images to be displayed per plot
        :param n_plots:                 Number of plots to display (each plot being a combination of img with bbox)
        :param plot_transformed_data:   If True, the plot will be over samples after applying transforms (i.e. on __getitem__).
                                        If False, the plot will be over the raw samples (i.e. on get_sample)
        :return:
        """
        plot_counter = 0
        input_format = self.output_target_format if plot_transformed_data else self.original_target_format
        target_format_transform = DetectionTargetsFormatTransform(input_format=input_format, output_format=DetectionTargetsFormat.XYXY_LABEL)

        for plot_i in range(n_plots):
            fig = plt.figure(figsize=(10, 10))
            n_subplot = int(np.ceil(max_samples_per_plot**0.5))
            for img_i in range(max_samples_per_plot):
                index = img_i + plot_i * 16

                if plot_transformed_data:
                    image, targets, *_ = self[img_i + plot_i * 16]
                    image = image.transpose(1, 2, 0).astype(np.int32)
                else:
                    sample = self.get_sample(index)
                    image, targets = sample["image"], sample["target"]

                sample = target_format_transform({"image": image, "target": targets})

                # shape = [padding_size x 4] (The dataset will most likely pad the targets to a fixed dim)
                boxes = sample["target"][:, 0:4]

                # shape = [n_box x 4] (We remove padded boxes, which corresponds to boxes with only 0)
                boxes = boxes[(boxes != 0).any(axis=1)]
                plt.subplot(n_subplot, n_subplot, img_i + 1).imshow(image[:, :, ::-1])
                plt.plot(boxes[:, [0, 2, 2, 0, 0]].T, boxes[:, [1, 1, 3, 3, 1]].T, ".-")
                plt.axis("off")
            fig.tight_layout()
            plt.show()
            plt.close()

            plot_counter += 1
            if plot_counter == n_plots:
                return
