import os
import json
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch

from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataSet
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST


_COCO_91_INDEX = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

_COCO_91_INVERTED_INDEX = {coco_id: index for index, coco_id in enumerate(_COCO_91_INDEX)}


def format_bbox(bbox: List[int], img_width: float, img_height: float):
    """
    :param bbox: Bounding box in format (x_left, y_top, width, height) not normalized
    :return:     Normalized bbox in centered and format (x_center, y_center, width, height) normalized between 0-1
    """
    result = [
        (bbox[0] + bbox[2] / 2) / img_width,
        (bbox[1] + bbox[3] / 2) / img_height,
        (bbox[2]) / img_width,
        (bbox[3]) / img_height,
    ]
    return [round(coordinate, 6) for coordinate in result]


class COCODetectionDataSet(DetectionDataSet):
    """
    COCODetectionDataSet - Detection Data Set Class COCO Data Set
    """

    def __init__(self, *args, **kwargs):
        kwargs['all_classes_list'] = COCO_DETECTION_CLASSES_LIST
        super().__init__(*args, **kwargs)

    def _load_additional_labels(self):
        annotations_json_path = os.path.join(self.root, self.additional_labels_path)
        with open(annotations_json_path) as file:
            anno_json = json.load(file)

        id_to_img = {img['id']: img for img in anno_json['images']}
        crowd_annotations = [annotation for annotation in anno_json['annotations'] if annotation['iscrowd'] == 1]

        img_id_to_crowd_gts = {annotation['image_id']: [] for annotation in crowd_annotations}
        for annotation in crowd_annotations:
            img = id_to_img[annotation['image_id']]
            img_id_to_crowd_gts[annotation['image_id']].append({
                'bbox': format_bbox(annotation['bbox'], img['width'], img['height']),
                'category_id': _COCO_91_INVERTED_INDEX[annotation['category_id']]
            })

        self.img_id_to_crowd_gts = img_id_to_crowd_gts
        self.img_ids = [int(Path(p).stem) for p in self.img_files]

    def _get_raw_crowd_labels(self, index: int) -> np.array:
        crowd_ground_truths = self.img_id_to_crowd_gts.get(self.img_ids[index], [])
        if len(crowd_ground_truths) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array([(gt['category_id'], *gt['bbox']) for gt in crowd_ground_truths])

    def _get_additional_labels(self, ratio, pad, index, preprocess_shape, postprocess_shape) -> Dict[str, torch.Tensor]:
        """Get the crowd labels in xywh format, normalized between 0 to 1. This
            :param ratio:             Image ratio
            :param pad:               Image padding
            :param index:             Image index
            :param preprocess_shape:  Image shape before preprocessing in format (height, weight)
            :param postprocess_shape: Image shape after preprocessing in format (height, weight)
            :return:                  Dict with the crowd labels for this image
        """
        prep_h, prep_w = preprocess_shape
        post_h, post_w = postprocess_shape
        crowd_labels = self._get_raw_crowd_labels(index)
        if len(crowd_labels) > 0:
            crowd_labels = self.target_transform(crowd_labels, ratio, prep_w, prep_h, pad)
            crowd_labels = self.convert_xyxy_bbox_to_normalized_xywh(crowd_labels, post_w, post_h)
        crowd_labels = self._convert_label_to_torch(crowd_labels)
        return {'crowd_target': crowd_labels}
