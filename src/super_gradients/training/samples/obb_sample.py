import dataclasses
from typing import Union, List, Optional

import numpy as np
import torch
from super_gradients.training.datasets.data_formats.obb.cxcywhr import cxcywhr_to_poly, poly_to_cxcywhr


@dataclasses.dataclass
class OBBSample:
    """
    A data class describing a single object detection sample that comes from a dataset.
    It contains both input image and target information to train an object detection model.

    :param image:              Associated image with a sample. Can be in [H,W,C] or [C,H,W] format
    :param rboxes_cxcywhr:      Numpy array of [N,5] shape with oriented bounding box of each instance (CX,CY,W,H,R)
    :param labels:             Numpy array of [N] shape with class label for each instance
    :param is_crowd:           (Optional) Numpy array of [N] shape with is_crowd flag for each instance
    :param additional_samples: (Optional) List of additional samples for the same image.
    """

    __slots__ = ["image", "rboxes_cxcywhr", "labels", "is_crowd", "additional_samples"]

    image: Union[np.ndarray, torch.Tensor]
    rboxes_cxcywhr: np.ndarray
    labels: np.ndarray
    is_crowd: np.ndarray
    additional_samples: Optional[List["OBBSample"]]

    def __init__(
        self,
        image: Union[np.ndarray, torch.Tensor],
        rboxes_cxcywhr: np.ndarray,
        labels: np.ndarray,
        is_crowd: Optional[np.ndarray] = None,
        additional_samples: Optional[List["OBBSample"]] = None,
    ):
        if is_crowd is None:
            is_crowd = np.zeros(len(labels), dtype=bool)

        if len(rboxes_cxcywhr) != len(labels):
            raise ValueError("Number of bounding boxes and labels must be equal. Got {len(bboxes_xyxy)} and {len(labels)} respectively")

        if len(rboxes_cxcywhr) != len(is_crowd):
            raise ValueError("Number of bounding boxes and is_crowd flags must be equal. Got {len(bboxes_xyxy)} and {len(is_crowd)} respectively")

        if len(rboxes_cxcywhr.shape) != 2 or rboxes_cxcywhr.shape[1] != 5:
            raise ValueError(f"Oriented boxes must be in [N,5] format. Shape of input bboxes is {rboxes_cxcywhr.shape}")

        if len(is_crowd.shape) != 1:
            raise ValueError(f"Number of is_crowd flags must be in [N] format. Shape of input is_crowd is {is_crowd.shape}")

        if len(labels.shape) != 1:
            raise ValueError("Labels must be in [N] format. Shape of input labels is {labels.shape}")

        self.image = image
        self.rboxes_cxcywhr = rboxes_cxcywhr
        self.labels = labels
        self.is_crowd = is_crowd
        self.additional_samples = additional_samples

    def sanitize_sample(self) -> "OBBSample":
        """
        Apply sanity checks on the detection sample, which includes clamping of rotate boxes to image boundaries
        and removing boxes with non-positive area.
        This method returns a new DetectionSample instance with sanitized data.
        :return: A DetectionSample after filtering.
        """
        polys = cxcywhr_to_poly(self.rboxes_cxcywhr)
        # Clamp polygons to image boundaries
        polys[..., 0] = np.clip(polys[..., 0], 0, self.image.shape[1])
        polys[..., 1] = np.clip(polys[..., 1], 0, self.image.shape[0])
        rboxes_cxcywhr = poly_to_cxcywhr(polys)
        return OBBSample(
            image=self.image,
            rboxes_cxcywhr=rboxes_cxcywhr,
            labels=self.labels,
            is_crowd=self.is_crowd,
            additional_samples=self.additional_samples,
        ).filter_by_bbox_area(0)

    def filter_by_mask(self, mask: np.ndarray) -> "OBBSample":
        """
        Remove boxes & labels with respect to a given mask.
        This method returns a new DetectionSample instance with filtered data.

        :param mask:   A boolean or integer mask of samples to keep for given sample.
        :return:       A DetectionSample after filtering.
        """
        return OBBSample(
            image=self.image,
            rboxes_cxcywhr=self.rboxes_cxcywhr[mask],
            labels=self.labels[mask],
            is_crowd=self.is_crowd[mask] if self.is_crowd is not None else None,
            additional_samples=self.additional_samples,
        )

    def filter_by_bbox_area(self, min_rbox_area: Union[int, float]) -> "OBBSample":
        """
        Remove pose instances that has area of the corresponding bounding box less than a certain threshold.

        :param min_rbox_area: Minimal rotated box area of the box to keep.
        :return:              A OBBSample after filtering.
        """
        area = self.rboxes_cxcywhr[..., 2:4].prod(axis=-1)
        keep_mask = area > min_rbox_area
        return self.filter_by_mask(keep_mask)

    def __len__(self):
        return len(self.labels)
