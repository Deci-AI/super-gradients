import typing
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import default_collate

from super_gradients.common.registry import register_collate_function

if typing.TYPE_CHECKING:
    from super_gradients.training.samples import DetectionSample


@register_collate_function()
class DetectionCollateFN:
    """
    A single collate function for training all detection models.
    The output targets format is [Batch, 6], where 6 is (batch_index, class_id, x, y, w, h)
    """

    def __init__(self):
        pass

    def __call__(self, batch: List["DetectionSample"]) -> Tuple[torch.Tensor, torch.Tensor, typing.Mapping]:
        from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import flat_collate_tensors_with_batch_index

        images = []
        targets = []

        for sample in batch:
            # Generate targets.
            # Here we use only non-crowd targets for training
            non_crowd_mask = sample.is_crowd == 0
            image_targets = np.concatenate((sample.labels[non_crowd_mask, None], sample.bboxes_xywh[non_crowd_mask]), axis=1)
            targets.append(torch.from_numpy(image_targets))

            # Convert image & mask to tensors
            # Change image layout from HWC to CHW
            sample.image = torch.from_numpy(np.transpose(sample.image, [2, 0, 1]))
            images.append(sample.image)

            # Remove image and mask from sample because at this point we don't need them anymore
            sample.image = None

            # Make sure additional samples are None, so they don't get collated as it causes collate to slow down
            sample.additional_samples = None

        images = default_collate(images)
        targets = flat_collate_tensors_with_batch_index(targets)

        extras = {"gt_samples": batch}
        return images, targets, extras
