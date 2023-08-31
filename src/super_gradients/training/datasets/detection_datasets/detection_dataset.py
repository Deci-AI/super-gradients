import collections
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

from super_gradients.common.object_names import Datasets, Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.module_interfaces import HasPreprocessingParams
from super_gradients.training.utils.detection_utils import get_class_index_in_target
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.transforms.transforms import DetectionTransform, DetectionTargetsFormatTransform, DetectionTargetsFormat
from super_gradients.common.exceptions.dataset_exceptions import EmptyDatasetException, DatasetValidationException
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
from super_gradients.training.datasets.data_formats.formats import ConcatenatedTensorFormat, LabelTensorSliceItem
from super_gradients.training.utils.utils import ensure_is_tuple_of_two

logger = get_logger(__name__)


@register_dataset(Datasets.DETECTION_DATASET)
class DetectionDataset(Dataset, HasPreprocessingParams):
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
        - Index:        Index of the sample in the dataset, AFTER filtering (if relevant). 0<=index<=len(dataset)-1
        - Sample ID:    Index of the sample in the dataset, WITHOUT considering any filtering. 0<=sample_id<=len(source)-1
    """

    @resolve_param("transforms", ListFactory(TransformsFactory()))
    def __init__(
        self,
        data_dir: str,
        original_target_format: Union[ConcatenatedTensorFormat, DetectionTargetsFormat],
        max_num_samples: int = None,
        cache: bool = False,
        cache_annotations: bool = True,
        cache_dir: str = None,
        input_dim: Union[int, Tuple[int, int], None] = None,
        transforms: List[DetectionTransform] = [],
        all_classes_list: Optional[List[str]] = [],
        class_inclusion_list: Optional[List[str]] = None,
        ignore_empty_annotations: bool = True,
        target_fields: List[str] = None,
        output_fields: List[str] = None,
        verbose: bool = True,
        show_all_warnings: bool = False,
    ):
        """Detection dataset.

        :param data_dir:                Where the data is stored
        :param input_dim:               Image size (when loaded, before transforms). Can be None, scalar or tuple (rows, cols).
                                        None means that the image will be loaded as is.
                                        Scalar (size) - Image will be resized to (size, size)
                                        Tuple (rows,cols) - Image will be resized to (rows, cols)
        :param original_target_format:  Format of targets stored on disk. raw data format, the output format might
                                        differ based on transforms.
        :param max_num_samples:         If not None, set the maximum size of the dataset by only indexing the first n annotations/images.
        :param cache:                   Whether to cache images or not.
        :param cache_annotations:       Whether to cache annotations or not. This reduces training time by pre-loading all the annotations,
                                        but requires more RAM and more time to instantiate the dataset when working on very large datasets.
        :param cache_dir:              Path to the directory where cached images will be stored in an optimized format.
        :param transforms:              List of transforms to apply sequentially on sample.
        :param all_classes_list:        All the class names.
        :param class_inclusion_list:    If not None, define the subset of classes to be included as targets.
                                        Classes not in this list will excluded from training.
                                        Thus, number of classes in model must be adjusted accordingly.
        :param ignore_empty_annotations:        If True and class_inclusion_list not None, images without any target
                                                will be ignored.
        :param target_fields:                   List of the fields target fields. This has to include regular target,
                                                but can also include crowd target, segmentation target, ...
                                                It has to include at least "target" but can include other.
        :param output_fields:                   Fields that will be outputed by __getitem__.
                                                It has to include at least "image" and "target" but can include other.
        :param verbose:                 Whether to show additional information or not, such as loading progress. (doesnt include warnings)
        :param show_all_warnings:       Whether to show all warnings or not.
        """
        super().__init__()
        self.verbose = verbose
        self.show_all_warnings = show_all_warnings

        if isinstance(original_target_format, DetectionTargetsFormat):
            logger.warning(
                "Deprecation: original_target_format should be of type ConcatenatedTensorFormat instead of DetectionTargetsFormat."
                "Support for DetectionTargetsFormat will be removed in 3.1"
            )

        self.data_dir = data_dir
        if not Path(data_dir).exists():
            raise RuntimeError(f"data_dir={data_dir} not found. Please make sure that data_dir points toward your dataset.")

        # Number of images that are available (regardless of ignored images)
        n_dataset_samples = self._setup_data_source()
        if not isinstance(n_dataset_samples, int) or n_dataset_samples < 1:
            raise ValueError(f"_setup_data_source() should return the number of available samples but got {n_dataset_samples}")
        n_samples = n_dataset_samples if max_num_samples is None else min(n_dataset_samples, max_num_samples)

        self.input_dim = ensure_is_tuple_of_two(input_dim)
        self.original_target_format = original_target_format

        if len(all_classes_list) != len(set(all_classes_list)):
            raise DatasetValidationException(f"all_classes_list contains duplicate class names: {collections.Counter(all_classes_list)}")

        if class_inclusion_list is not None and len(class_inclusion_list) != len(set(class_inclusion_list)):
            raise DatasetValidationException(f"class_inclusion_list contains duplicate class names: {collections.Counter(class_inclusion_list)}")

        self.all_classes_list = all_classes_list or self._all_classes
        self.class_inclusion_list = class_inclusion_list
        self.classes = self.class_inclusion_list or self.all_classes_list
        if len(set(self.classes) - set(self.all_classes_list)) > 0:
            wrong_classes = set(self.classes) - set(all_classes_list)
            raise DatasetValidationException(
                f"{wrong_classes} defined in `class_inclusion_list` were not found among `all_classes_list={self.all_classes_list}`"
            )

        self.ignore_empty_annotations = ignore_empty_annotations
        self.target_fields = target_fields or ["target"]
        if "target" not in self.target_fields:
            raise KeyError('"target" is expected to be in the fields to subclass but it was not included')

        self._required_annotation_fields = {"target", "img_path", "resized_img_shape"}

        self.transforms = transforms

        self.output_fields = output_fields or ["image", "target"]
        if len(self.output_fields) < 2 or self.output_fields[0] != "image" or self.output_fields[1] != "target":
            raise ValueError('output_fields must start with "image" and then "target", followed by any other field')

        self._cache_annotations = cache_annotations
        self._cached_annotations: Dict[int, Dict] = {}  # We use a dict and not a list because when `ignore_empty_annotations=True` we may ignore some indexes.

        # Maps (dataset index) -> (non-empty sample ids)
        self._non_empty_sample_ids: Optional[List[int]] = None

        # Some transform may require non-empty annotations to be indexed.
        transform_require_non_empty_annotations = any(getattr(transform, "non_empty_annotations", False) for transform in self.transforms)

        # Iterate over the whole dataset to index the images with/without annotations.
        if self._cache_annotations or self.ignore_empty_annotations or transform_require_non_empty_annotations:

            if self._cache_annotations:
                logger.info("Dataset Initialization in progress. `cache_annotations=True` causes the process to take longer due to full dataset indexing.")
            elif self.ignore_empty_annotations:
                logger.info(
                    "Dataset Initialization in progress. `ignore_empty_annotations=True` causes the process to take longer due to full dataset indexing."
                )
            elif transform_require_non_empty_annotations:
                logger.info(
                    "Dataset Initialization in progress. "
                    "Having a transform with `non_empty_annotations=True` set causes the process to take longer due to the need for a full dataset indexing."
                )

            # Map indexes to sample annotations.
            non_empty_annotations, empty_annotations = self._load_all_annotations(n_samples=n_samples)
            if self._cache_annotations:
                if self.ignore_empty_annotations and transform_require_non_empty_annotations:
                    self._cached_annotations = non_empty_annotations
                else:
                    # Non overlapping dicts. since they map unique sample_ids -> sample
                    self._cached_annotations = {**non_empty_annotations, **empty_annotations}

            if self.ignore_empty_annotations and len(non_empty_annotations) == 0:
                raise EmptyDatasetException(f"Out of {n_samples} images, not a single one was found with any of these classes: {self.class_inclusion_list}")

            self._non_empty_sample_ids = list(non_empty_annotations.keys())

        self._n_samples = n_samples  # Regardless of any filtering

        # CACHE IMAGE
        self.cache = cache
        self.cache_dir = cache_dir
        self.cached_imgs_padded = self._cache_images() if self.cache else None

    @property
    def _all_classes(self):
        """Placeholder to setup the class names. This is an alternative to passing "all_classes_list" to __init__.
        This is usefull when all_classes_list is not known in advance, only after loading the dataset."""
        raise NotImplementedError

    def _setup_data_source(self) -> int:
        """Set up the data source and store relevant objects as attributes.

        :return: Number of available samples, (i.e. how many images we have, regardless of any filter we might want to use)"""
        raise NotImplementedError

    def _load_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Load annotations associated to a specific sample.
        Please note that the targets should be resized according to self.input_dim!

        :param sample_id:   Sample ID refers to the index of the sample in the dataset, WITHOUT considering any filtering. 0<=sample_id<=len(source)-1
        :return:            Annotation, a dict with any field but has to include at least the fields specified in self._required_annotation_fields.
        """
        raise NotImplementedError

    def _get_sample_annotations(self, index: int, ignore_empty_annotations: bool) -> Dict[str, Union[np.ndarray, Any]]:
        """Get the annotation associated to a specific sample. Use cache if enabled.
        :param index:                       Index refers to the index of the sample in the dataset, AFTER filtering (if relevant). 0<=index<=len(dataset)-1
        :param ignore_empty_annotations:    Whether to ignore empty annotations or not.
        :return:                            Dict representing the annotation of a specific image
        """
        sample_id = self._non_empty_sample_ids[index] if ignore_empty_annotations else index
        if self._cache_annotations:
            return self._cached_annotations[sample_id]
        else:
            return self._load_sample_annotation(sample_id=sample_id)

    def _load_sample_annotation(self, sample_id: int) -> Dict[str, Union[np.ndarray, Any]]:
        """Load the annotation associated to a specific sample and apply subclassing.
        :param sample_id:   Sample ID refers to the index of the sample in the dataset, WITHOUT considering any filtering. 0<=sample_id<=len(source)-1
        """
        sample_annotations = self._load_annotation(sample_id=sample_id)
        if not self._required_annotation_fields.issubset(set(sample_annotations.keys())):
            raise KeyError(
                f"_load_annotation is expected to return at least the fields {self._required_annotation_fields}, but got {set(sample_annotations.keys())}"
            )

        # Filter out classes that are not in self.class_inclusion_list
        if self.class_inclusion_list is not None:
            sample_annotations = self._sub_class_annotation(annotation=sample_annotations)
        return sample_annotations

    def _load_all_annotations(self, n_samples: int) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """Load ALL the annotations into memory. This is usually required when `ignore_empty_annotations=True`,
        because we have to iterate over the whole dataset once in order to know which sample is empty and which is not.
        Question: Why not just check if annotation is empty on the fly ?
        Answer: When running with DDP, we split the dataset into small chunks.
                Therefore, we need to make sure that each chunk includes a similar subset of index.
                If we were to check on the fly, we would not know in advance the size of dataset/chunks
                and this means that some chunks would be smaller than others

        :param n_samples:   Number of samples in the datasets (including samples without annotations).
        :return:            A tuple of two dicts, one for non-empty annotations and one for empty annotations
                                - non_empty_annotations: Dict mapping dataset index -> non-empty annotations
                                - empty_annotations:     Dict mapping dataset index -> empty annotations
        """
        n_invalid_bbox = 0
        non_empty_annotations, empty_annotations = {}, {}

        for index in tqdm(range(n_samples), desc="Indexing dataset annotations", disable=not self.verbose):

            sample_annotations = self._load_sample_annotation(sample_id=index)
            n_invalid_bbox += sample_annotations.get("n_invalid_labels", 0)

            is_annotation_non_empty = any(len(sample_annotations[field]) != 0 for field in self.target_fields)
            if is_annotation_non_empty:
                non_empty_annotations[index] = sample_annotations if self._cache_annotations else None
            else:
                empty_annotations[index] = sample_annotations if self._cache_annotations else None

        if len(non_empty_annotations) + len(empty_annotations) == 0:
            raise EmptyDatasetException(f"Out of {n_samples} images, not a single one was found with any of these classes: {self.class_inclusion_list}")

        if n_invalid_bbox > 0:
            logger.warning(f"Found {n_invalid_bbox} invalid bbox that were ignored. For more information, please set `show_all_warnings=True`.")

        return non_empty_annotations, empty_annotations

    def _sub_class_annotation(self, annotation: dict) -> Union[dict, None]:
        """Subclass every field listed in self.target_fields. It could be targets, crowd_targets, ...

        :param annotation: Dict representing the annotation of a specific image
        :return:           Subclassed annotation if non-empty after subclassing, otherwise None
        """
        class_index = _get_class_index_in_target(target_format=self.original_target_format)
        for field in self.target_fields:
            annotation[field] = self._sub_class_target(targets=annotation[field], class_index=class_index)
        return annotation

    def _sub_class_target(self, targets: np.ndarray, class_index: int) -> np.ndarray:
        """Sublass targets of a specific image.

        :param targets:     Target array to subclass of shape [n_targets, 5], 5 representing a bbox
        :param class_index:    Position of the class id in a bbox
                                ex: 0 if bbox of format label_xyxy | -1 if bbox of format xyxy_label
        :return:            Subclassed target
        """
        targets_kept = []
        for target in targets:
            cls_id = int(target[class_index])
            cls_name = self.all_classes_list[cls_id]
            if cls_name in self.class_inclusion_list:
                # Replace the target cls_id in self.all_classes_list by cls_id in self.class_inclusion_list
                target[class_index] = self.class_inclusion_list.index(cls_name)
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

        if self.input_dim is None:
            raise RuntimeError("caching is not possible without input_dim is not set")
        max_h, max_w = self.input_dim[0], self.input_dim[1]

        # The cache should be the same as long as the images and their sizes are the same
        hash = hashlib.sha256()
        for index in range(len(self)):
            annotation = self._get_sample_annotations(index=index, ignore_empty_annotations=self.ignore_empty_annotations)
            values_to_hash = [annotation["resized_img_shape"][0], annotation["resized_img_shape"][1], Path(annotation["img_path"]).name]
            for value in values_to_hash:
                hash.update(str(value).encode("utf-8"))
        cache_hash = hash.hexdigest()

        img_resized_cache_path = cache_dir / f"img_resized_cache_{cache_hash}.array"

        if not img_resized_cache_path.exists():
            logger.info("Caching images for the first time. Be aware that this will stay in the disk until you delete it yourself.")
            NUM_THREADs = min(8, os.cpu_count())

            # Inline-function because we should not to pollute the rest of the class with this function.
            # This function is required because of legacy design - ideally we should not have to load annotations in order to get the image path.
            def _load_image_from_index(index: int) -> np.ndarray:
                annotations = self._get_sample_annotations(index=index, ignore_empty_annotations=self.ignore_empty_annotations)
                return self._load_resized_img(image_path=annotations["img_path"])

            loaded_images = ThreadPool(NUM_THREADs).imap(func=_load_image_from_index, iterable=range(len(self)))

            # Initialize placeholder for images
            cached_imgs = np.memmap(str(img_resized_cache_path), shape=(len(self), max_h, max_w, 3), dtype=np.uint8, mode="w+")

            # Store images in the placeholder
            with tqdm(enumerate(loaded_images), total=len(self), disable=not self.verbose) as loaded_images_pbar:
                for i, image in loaded_images_pbar:
                    cached_imgs[i][: image.shape[0], : image.shape[1], :] = image.copy()
                cached_imgs.flush()
        else:
            logger.warning("You are using cached imgs!")

        logger.info("Loading cached imgs...")
        cached_imgs = np.memmap(str(img_resized_cache_path), shape=(len(self), max_h, max_w, 3), dtype=np.uint8, mode="r+")
        return cached_imgs

    def _load_resized_img(self, image_path: str) -> np.ndarray:
        """Load an image and resize it to the desired size (If relevant).
        :param image_path:  Full path of the image
        :return:            Image in BGR format, and channel last (HWC).
        """
        img = self._load_image(image_path=image_path)

        if self.input_dim is not None:
            r = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
            desired_size = (int(img.shape[1] * r), int(img.shape[0] * r))
            img = cv2.resize(src=img, dsize=desired_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        return img

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image.
        :param image_path:  Full path of the image
        :return:            Image in BGR format, and channel last (HWC).
        """
        img_file = os.path.join(image_path)
        img = cv2.imread(img_file)

        if img is None:
            raise FileNotFoundError(f"{img_file} was no found. Please make sure that the dataset was" f"downloaded and that the path is correct")
        return img

    def __del__(self):
        """Clear the cached images"""
        if hasattr(self, "cached_imgs_padded"):
            del self.cached_imgs_padded

    def __len__(self) -> int:
        """Get the length of the dataset. Note that this is the number of samples AFTER filtering (if relevant)."""
        return len(self._non_empty_sample_ids) if self.ignore_empty_annotations else self._n_samples

    def __getitem__(self, index: int) -> Tuple:
        """Get the sample post transforms at a specific index of the dataset.
        The output of this function will be collated to form batches.

        :param index:   Index refers to the index of the sample in the dataset, AFTER filtering (if relevant). 0<=index<=len(dataset)-1
        :return:        Sample, i.e. a dictionary including at least "image" and "target"
        """
        sample = self.apply_transforms(self.get_sample(index=index, ignore_empty_annotations=self.ignore_empty_annotations))
        for field in self.output_fields:
            if field not in sample.keys():
                raise KeyError(f"The field {field} must be present in the sample but was not found." "Please check the output fields of your transforms.")
        return tuple(sample[field] for field in self.output_fields)

    def get_random_item(self):
        return self[self.get_random_sample(ignore_empty_annotations=self.ignore_empty_annotations)]

    def get_sample(self, index: int, ignore_empty_annotations: bool = False) -> Dict[str, Union[np.ndarray, Any]]:
        """Get raw sample, before any transform (beside subclassing).
        :param index:                       Index refers to the index of the sample in the dataset, AFTER filtering (if relevant). 0<=index<=len(dataset)-1
        :param ignore_empty_annotations:    If True, empty annotations will be ignored
        :return:                            Sample, i.e. a dictionary including at least "image" and "target"
        """
        sample_annotations = self._get_sample_annotations(index=index, ignore_empty_annotations=ignore_empty_annotations)
        if self.cache:
            image = self._get_cached_image(index=index, cached_image_shape=sample_annotations["resized_img_shape"])
        else:
            image = self._load_resized_img(image_path=sample_annotations["img_path"])
        return {"image": image, **deepcopy(sample_annotations)}

    def _get_cached_image(self, index: int, cached_image_shape: Tuple[int, int]) -> np.ndarray:
        """Load an image from cache.
        :param index:               Index refers to the index of the sample in the dataset, AFTER filtering (if relevant). 0<=index<=len(dataset)-1
        :param cached_image_shape:  Shape of the cached image (after resizing if input_dim is set)
        :return:                    Image
        """
        padded_image = self.cached_imgs_padded[index]
        cached_height, cached_width = cached_image_shape
        resized_image = padded_image[:cached_height, :cached_width, :]
        return resized_image.copy()

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
            sample["additional_samples"] = self._get_additional_inputs_for_transform(transform=transform)
            sample = transform(sample=sample)
            sample.pop("additional_samples")  # additional_samples is not useful after the transform
        return sample

    def _get_additional_inputs_for_transform(self, transform: DetectionTransform) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Add additional inputs required by a transform to the sample"""
        additional_samples_count = transform.additional_samples_count if hasattr(transform, "additional_samples_count") else 0
        non_empty_annotations = transform.non_empty_annotations if hasattr(transform, "non_empty_annotations") else False
        return self.get_random_samples(count=additional_samples_count, ignore_empty_annotations=non_empty_annotations)

    def get_random_samples(self, count: int, ignore_empty_annotations: bool = False) -> List[Dict[str, Union[np.ndarray, Any]]]:
        """Load random samples.

        :param count: The number of samples wanted
        :param ignore_empty_annotations: If true, only return samples with at least 1 annotation
        :return: A list of samples satisfying input params
        """
        return [self.get_random_sample(ignore_empty_annotations) for _ in range(count)]

    def get_random_sample(self, ignore_empty_annotations: bool = False) -> Dict[str, Union[np.ndarray, Any]]:
        n_relevant_samples = len(self._non_empty_sample_ids) if ignore_empty_annotations else self._n_samples
        random_index = random.randint(0, n_relevant_samples - 1)
        return self.get_sample(index=random_index, ignore_empty_annotations=ignore_empty_annotations)

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
        if isinstance(input_format, DetectionTargetsFormat):
            raise ValueError(
                "Plot is not supported for DetectionTargetsFormat. Please set original_target_format to be an isntance of ConcatenateTransform instead."
            )
        target_format_transform = DetectionTargetsFormatTransform(input_format=input_format, output_format=XYXY_LABEL)

        for plot_i in range(n_plots):
            fig = plt.figure(figsize=(10, 10))
            n_subplot = int(np.ceil(max_samples_per_plot**0.5))
            for img_i in range(max_samples_per_plot):
                index = img_i + plot_i * 16

                if plot_transformed_data:
                    image, targets, *_ = self[img_i + plot_i * 16]
                    image = image.transpose(1, 2, 0).astype(np.int32)
                else:
                    sample = self.get_sample(index=index, ignore_empty_annotations=self.ignore_empty_annotations)
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

    def get_dataset_preprocessing_params(self):
        """
        Return any hardcoded preprocessing + adaptation for PIL.Image image reading (RGB).
         image_processor as returned as as list of dicts to be resolved by processing factory.
        :return:
        """
        pipeline = [Processings.ReverseImageChannels]
        if self.input_dim is not None:
            pipeline += [{Processings.DetectionLongestMaxSizeRescale: {"output_shape": self.input_dim}}]
        for t in self.transforms:
            pipeline += t.get_equivalent_preprocessing()
        params = dict(
            class_names=self.classes,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            iou=0.65,
            conf=0.5,
        )
        return params


def _get_class_index_in_target(target_format: DetectionTargetsFormat) -> int:
    """Get the index of the class in the target format.
    :param target_format: format of the target. E.g. XYXY_LABEL, LABEL_NORMALIZED_XYXY, ect...
    :return: index of the class in the target format. E.g. XYXY_LABEL -> 4, LABEL_NORMALIZED_XYXY -> 0, ect....
    """
    if isinstance(target_format, ConcatenatedTensorFormat):
        return target_format.indexes[LabelTensorSliceItem.NAME][0]
    elif isinstance(target_format, DetectionTargetsFormat):
        return get_class_index_in_target(target_format)
    else:
        raise NotImplementedError(
            f"{target_format} is not supported. Supported formats are: {ConcatenatedTensorFormat.__name__}, {DetectionTargetsFormat.__name__}"
        )
