import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from typing import List, Optional

from contextlib import redirect_stdout
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.exceptions.dataset_exceptions import DatasetValidationException, ParameterMismatchException
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL

logger = get_logger(__name__)


class COCOFormatDetectionDataset(DetectionDataset):
    """Base dataset to load ANY dataset that is with a similar structure to the COCO dataset.
    - Annotation file (.json). It has to respect the exact same format as COCO, for both the json schema and the bbox format (xywh).
    - One folder with all the images.

    Output format: (x, y, x, y, class_id)
    """

    def __init__(
        self,
        data_dir: str,
        json_annotation_file: str,
        images_dir: str,
        tight_box_rotation: bool = False,
        with_crowd: bool = True,
        class_ids_to_ignore: Optional[List[int]] = None,
        *args,
        **kwargs,
    ):
        """
        :param data_dir:                Where the data is stored.
        :param json_annotation_file:    Name of the coco json file. Path relative to data_dir.
        :param images_dir:              Name of the directory that includes all the images. Path relative to data_dir.
        :param tight_box_rotation:      bool, whether to use of segmentation maps convex hull as target_seg
                                            (check get_sample docs).
        :param with_crowd:              Add the crowd groundtruths to __getitem__
        :param class_ids_to_ignore:     List of class ids to ignore in the dataset. By default, doesnt ignore any class.
        """
        self.images_dir = images_dir
        self.json_annotation_file = json_annotation_file
        self.tight_box_rotation = tight_box_rotation
        self.with_crowd = with_crowd
        self.class_ids_to_ignore = class_ids_to_ignore or []

        target_fields = ["target", "crowd_target"] if self.with_crowd else ["target"]
        kwargs["target_fields"] = target_fields
        kwargs["output_fields"] = ["image", *target_fields]
        kwargs["original_target_format"] = XYXY_LABEL
        super().__init__(data_dir=data_dir, *args, **kwargs)

        if len(self.original_classes) != len(self.all_classes_list):
            if set(self.all_classes_list).issubset(set(self.original_classes)):
                raise ParameterMismatchException(
                    "Parameter `all_classes_list` contains a subset of classes from dataset JSON. "
                    "Please use `class_inclusion_list` to train with reduced number of classes",
                )
            else:
                raise DatasetValidationException(
                    "Number of classes in dataset JSON do not match with number of classes in all_classes_list parameter. "
                    "Most likely this indicates an error in your all_classes_list parameter"
                )

    def _setup_data_source(self) -> int:
        """Initialize img_and_target_path_list and warn if label file is missing

        :return: List of tuples made of (img_path,target_path)
        """

        self.coco = self._init_coco()
        self.class_ids = sorted(cls_id for cls_id in self.coco.getCatIds() if cls_id not in self.class_ids_to_ignore)
        self.original_classes = list([category["name"] for category in self.coco.loadCats(self.class_ids)])
        self.classes = copy.deepcopy(self.original_classes)
        self.sample_id_to_coco_id = self.coco.getImgIds()
        return len(self.sample_id_to_coco_id)

    @property
    def _all_classes(self) -> List[str]:
        return self.original_classes

    def _init_coco(self) -> COCO:
        annotation_file_path = os.path.join(self.data_dir, self.json_annotation_file)
        if not os.path.exists(annotation_file_path):
            raise ValueError("Could not find annotation file under " + str(annotation_file_path))

        if not self.verbose:
            with redirect_stdout(open(os.devnull, "w")):
                coco = COCO(annotation_file_path)
        else:
            coco = COCO(annotation_file_path)

        remove_useless_info(coco, self.tight_box_rotation)
        return coco

    def _load_annotation(self, sample_id: int) -> dict:
        """
        Load relevant information of a specific image.

        :param sample_id:               Sample_id in the dataset
        :return target:                 Target Bboxes (detection) in XYXY_LABEL format
        :return crowd_target:           Crowd target Bboxes (detection) in XYXY_LABEL format
        :return target_segmentation:    Segmentation
        :return initial_img_shape:      Image (height, width)
        :return resized_img_shape:      Resides image (height, width)
        :return img_path:               Path to the associated image
        """

        img_id = self.sample_id_to_coco_id[sample_id]

        img_metadata = self.coco.loadImgs(img_id)[0]
        width = img_metadata["width"]
        height = img_metadata["height"]

        img_annotation_ids = self.coco.getAnnIds(imgIds=[int(img_id)])
        img_annotations = self.coco.loadAnns(img_annotation_ids)

        cleaned_annotations = []
        for annotation in img_annotations:
            x1 = np.max((0, annotation["bbox"][0]))
            y1 = np.max((0, annotation["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, annotation["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, annotation["bbox"][3]))))
            if annotation["area"] > 0 and x2 >= x1 and y2 >= y1:
                annotation["clean_bbox"] = [x1, y1, x2, y2]
                cleaned_annotations.append(annotation)

        non_crowd_annotations = [annotation for annotation in cleaned_annotations if annotation["iscrowd"] == 0]

        target = np.zeros((len(non_crowd_annotations), 5))
        num_seg_values = 98 if self.tight_box_rotation else 0
        target_segmentation = np.ones((len(non_crowd_annotations), num_seg_values))
        target_segmentation.fill(np.nan)
        for ix, annotation in enumerate(non_crowd_annotations):
            cls = self.class_ids.index(annotation["category_id"])
            target[ix, 0:4] = annotation["clean_bbox"]
            target[ix, 4] = cls
            if self.tight_box_rotation:
                seg_points = [j for i in annotation.get("segmentation", []) for j in i]
                if seg_points:
                    seg_points_c = np.array(seg_points).reshape((-1, 2)).astype(np.int)
                    seg_points_convex = cv2.convexHull(seg_points_c).ravel()
                else:
                    seg_points_convex = []
                target_segmentation[ix, : len(seg_points_convex)] = seg_points_convex

        crowd_annotations = [annotation for annotation in cleaned_annotations if annotation["iscrowd"] == 1]

        crowd_target = np.zeros((len(crowd_annotations), 5))
        for ix, annotation in enumerate(crowd_annotations):
            cls = self.class_ids.index(annotation["category_id"])
            crowd_target[ix, 0:4] = annotation["clean_bbox"]
            crowd_target[ix, 4] = cls

        # Currently, the base class includes a feature to resize the image, so we need to resize the target as well when self.input_dim is set.
        initial_img_shape = (height, width)
        if self.input_dim is not None:
            r = min(self.input_dim[0] / height, self.input_dim[1] / width)
            target[:, :4] *= r
            crowd_target[:, :4] *= r
            target_segmentation *= r
            resized_img_shape = (int(height * r), int(width * r))
        else:
            resized_img_shape = initial_img_shape

        file_name = img_metadata["file_name"] if "file_name" in img_metadata else "{:012}".format(img_id) + ".jpg"
        img_path = os.path.join(self.data_dir, self.images_dir, file_name)
        img_id = self.sample_id_to_coco_id[sample_id]

        annotation = {
            "target": target,
            "crowd_target": crowd_target,
            "target_segmentation": target_segmentation,
            "initial_img_shape": initial_img_shape,
            "resized_img_shape": resized_img_shape,
            "img_path": img_path,
            "id": np.array([img_id]),
        }
        return annotation


def remove_useless_info(coco: COCO, use_seg_info: bool = False) -> None:
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset and not use_seg_info:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)
