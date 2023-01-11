import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataset
from super_gradients.training.exceptions.dataset_exceptions import DatasetValidationException, ParameterMismatchException
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat

logger = get_logger(__name__)


class COCODetectionDataset(DetectionDataset):
    """Dataset for COCO object detection."""

    def __init__(
        self,
        json_file: str = "instances_train2017.json",
        subdir: str = "images/train2017",
        tight_box_rotation: bool = False,
        with_crowd: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param json_file:           Name of the coco json file, that resides in data_dir/annotations/json_file.
        :param subdir:              Sub directory of data_dir containing the data.
        :param tight_box_rotation:  bool, whether to use of segmentation maps convex hull as target_seg
                                    (check get_sample docs).
        :param with_crowd: Add the crowd groundtruths to __getitem__

        kwargs:
            all_classes_list: all classes list, default is COCO_DETECTION_CLASSES_LIST.
        """
        self.subdir = subdir
        self.json_file = json_file
        self.tight_box_rotation = tight_box_rotation
        self.with_crowd = with_crowd

        target_fields = ["target", "crowd_target"] if self.with_crowd else ["target"]
        kwargs["target_fields"] = target_fields
        kwargs["output_fields"] = ["image", *target_fields]
        kwargs["original_target_format"] = DetectionTargetsFormat.XYXY_LABEL
        kwargs["all_classes_list"] = kwargs.get("all_classes_list") or COCO_DETECTION_CLASSES_LIST
        super().__init__(*args, **kwargs)

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
        self.class_ids = sorted(self.coco.getCatIds())
        self.original_classes = list([category["name"] for category in self.coco.loadCats(self.class_ids)])
        self.classes = copy.deepcopy(self.original_classes)
        self.sample_id_to_coco_id = self.coco.getImgIds()
        return len(self.sample_id_to_coco_id)

    def _init_coco(self) -> COCO:
        annotation_file_path = os.path.join(self.data_dir, "annotations", self.json_file)
        if not os.path.exists(annotation_file_path):
            raise ValueError("Could not find annotation file under " + str(annotation_file_path))

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

        r = min(self.input_dim[0] / height, self.input_dim[1] / width)
        target[:, :4] *= r
        crowd_target[:, :4] *= r
        target_segmentation *= r

        initial_img_shape = (height, width)
        resized_img_shape = (int(height * r), int(width * r))

        file_name = img_metadata["file_name"] if "file_name" in img_metadata else "{:012}".format(img_id) + ".jpg"
        img_path = os.path.join(self.data_dir, self.subdir, file_name)
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


def remove_useless_info(coco, use_seg_info=False):
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
