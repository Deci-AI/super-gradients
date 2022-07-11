import os
import json
from typing import Dict
from pathlib import Path

import numpy as np
import torch

from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionDataSet
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST, COCO_DEFAULT_CLASSES_TUPLES_LIST
from super_gradients.training.utils.detection_utils import convert_xywh_to_cxcywh


class COCODetectionDataSet(DetectionDataSet):
    """
    COCODetectionDataSet - Detection Data Set Class COCO Data Set
    """

    def __init__(self, *args, **kwargs):
        kwargs['all_classes_list'] = COCO_DETECTION_CLASSES_LIST
        super().__init__(*args, **kwargs)

    def _load_additional_labels(self):
        """
        Load all the crowd targets and store it in self.img_id_to_crowd_gts for later use.
        """

        coco_detection_category_id_list = [
            category_id
            for category_id, category_name in COCO_DEFAULT_CLASSES_TUPLES_LIST
            if category_name in COCO_DETECTION_CLASSES_LIST
        ]

        category_id_to_annotation_id = {
            category_id: annotation_id for annotation_id, category_id in enumerate(coco_detection_category_id_list)}

        annotations_json_path = os.path.join(self.root, self.additional_labels_path)
        with open(annotations_json_path) as file:
            anno_json = json.load(file)

        id_to_img = {img['id']: img for img in anno_json['images']}
        crowd_annotations = [annotation for annotation in anno_json['annotations'] if annotation['iscrowd'] == 1]

        img_id_to_crowd_gts = {annotation['image_id']: [] for annotation in crowd_annotations}
        for annotation in crowd_annotations:
            img = id_to_img[annotation['image_id']]
            img_id_to_crowd_gts[annotation['image_id']].append({
                'bbox': convert_xywh_to_cxcywh(annotation['bbox'], img['width'], img['height']),
                'category_id': category_id_to_annotation_id[annotation['category_id']]
            })

        self.img_id_to_crowd_gts = img_id_to_crowd_gts
        self.img_ids = [int(Path(p).stem) for p in self.img_files]

    def _get_raw_crowd_labels(self, index: int) -> np.array:
        """
        Get the crowd labels of a given image before any processing
            :param index:             Image index
            :return:                  Array representing the crowd labels
                                        format (label, x, y, w, h) where x,y,w,h are in range [0,1]
        """
        crowd_ground_truths = self.img_id_to_crowd_gts.get(self.img_ids[index], [])
        if len(crowd_ground_truths) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array([(gt['category_id'], *gt['bbox']) for gt in crowd_ground_truths])

    def _get_additional_labels(self, ratio, pad, index, preprocess_shape, postprocess_shape) -> Dict[str, torch.Tensor]:
        """
        Get the crowd labels in xywh format, normalized between 0 to 1
            :param ratio:             Image ratio
            :param pad:               Image padding
            :param index:             Image index
            :param preprocess_shape:  Image shape before preprocessing in format (height, weight)
            :param postprocess_shape: Image shape after preprocessing in format (height, weight)
            :return:                  Dict with the crowd labels for this image
                                        format (label, x, y, w, h) where x,y,w,h are in range [0,1]
        """
        prep_h, prep_w = preprocess_shape
        post_h, post_w = postprocess_shape
        crowd_labels = self._get_raw_crowd_labels(index)
        if len(crowd_labels) > 0:
            crowd_labels = self.target_transform(crowd_labels, ratio, prep_w, prep_h, pad)
            crowd_labels = self.convert_xyxy_bbox_to_normalized_xywh(crowd_labels, post_w, post_h)
        crowd_labels = self._convert_label_to_torch(crowd_labels)
        return {'crowd_targets': crowd_labels}
