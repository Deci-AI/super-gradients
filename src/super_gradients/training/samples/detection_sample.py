import dataclasses
import numpy as np
import torch

from typing import Optional, List, Union

from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh

__all__ = ["DetectionSample"]


@dataclasses.dataclass
class DetectionSample:
    """
    A data class describing a single object detection sample that comes from a dataset.
    It contains both input image and target information to train an object detection model.

    :param image:              Associated image with a sample. Can be in [H,W,C] or [C,H,W] format
    :param bboxes_xywh:        Numpy array of [N,4] shape with bounding box of each instance (XYWH)
    :param labels:             Numpy array of [N] shape with class label for each instance
    :param is_crowd:           (Optional) Numpy array of [N] shape with is_crowd flag for each instance
    :param additional_samples: (Optional) List of additional samples for the same image.
    """

    __slots__ = ["image", "bboxes_xywh", "labels", "is_crowd", "additional_samples"]

    image: Union[np.ndarray, torch.Tensor]
    bboxes_xywh: np.ndarray
    labels: np.ndarray
    is_crowd: Optional[np.ndarray]
    additional_samples: Optional[List["DetectionSample"]]

    def sanitize_sample(self) -> "DetectionSample":
        """
        Apply sanity checks on the detection sample, which includes clamping of bounding boxes to image boundaries.
        This function does not remove instances, but may make them subject for removal later on.
        This method operates in-place and modifies the caller.
        :return: A DetectionSample after filtering (caller instance).
        """
        image_height, image_width, _ = self.image.shape

        # Clamp bboxes to image boundaries
        clamped_boxes = xywh_to_xyxy(self.bboxes_xywh, image_shape=(image_height, image_width))
        clamped_boxes[..., [0, 2]] = np.clip(clamped_boxes[..., [0, 2]], 0, image_width - 1)
        clamped_boxes[..., [1, 3]] = np.clip(clamped_boxes[..., [1, 3]], 0, image_height - 1)
        clamped_boxes = xyxy_to_xywh(clamped_boxes, image_shape=(image_height, image_width))

        self.bboxes_xywh = clamped_boxes
        return self

    def filter_by_mask(self, mask: np.ndarray) -> "DetectionSample":
        """
        Remove boxes & labels with respect to a given mask.
        This method operates in-place and modifies the caller.
        If you are implementing a subclass of DetectionSample and adding extra field associated with each bbox
        instance (Let's say you add a distance property for each bbox from the camera), then you should override
        this method to do filtering on extra attribute as well.

        :param mask:   A boolean or integer mask of samples to keep for given sample.
        :return:       A DetectionSample after filtering (caller instance).
        """
        self.bboxes_xywh = self.bboxes_xywh[mask]
        self.labels = self.labels[mask]
        if self.is_crowd is not None:
            self.is_crowd = self.is_crowd[mask]
        return self

    def filter_by_bbox_area(self, min_bbox_area: Union[int, float]) -> "DetectionSample":
        """
        Remove pose instances that has area of the corresponding bounding box less than a certain threshold.
        This method operates in-place and modifies the caller.

        :param min_bbox_area: Minimal bounding box area of the pose to keep.
        :return:              A DetectionSample after filtering (caller instance).
        """
        area = self.bboxes_xywh[..., 2:4].prod(axis=-1)
        keep_mask = area >= min_bbox_area
        return self.filter_by_mask(keep_mask)
