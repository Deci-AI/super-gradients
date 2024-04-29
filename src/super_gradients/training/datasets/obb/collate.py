from typing import List

import numpy as np
import torch
from super_gradients.common.registry import register_collate_function

from .sample import OBBSample


@register_collate_function()
class OrientedBoxesCollate:
    def __call__(self, batch: List[OBBSample]):
        from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import flat_collate_tensors_with_batch_index

        images = []
        all_boxes = []
        all_labels = []
        all_crowd_masks = []

        for sample in batch:
            images.append(torch.from_numpy(np.transpose(sample.image, [2, 0, 1])))
            all_boxes.append(torch.from_numpy(sample.rboxes_cxcywhr))
            all_labels.append(torch.from_numpy(sample.labels.reshape((-1, 1))))
            all_crowd_masks.append(torch.from_numpy(sample.is_crowd.reshape((-1, 1))))
            sample.image = None

        images = torch.stack(images)

        boxes = flat_collate_tensors_with_batch_index(all_boxes).float()
        labels = flat_collate_tensors_with_batch_index(all_labels).long()
        is_crowd = flat_collate_tensors_with_batch_index(all_crowd_masks)

        extras = {"gt_samples": batch}
        return images, (boxes, labels, is_crowd), extras
