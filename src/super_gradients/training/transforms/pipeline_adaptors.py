from enum import Enum
from typing import Callable
from abc import abstractmethod, ABC
import numpy as np
from PIL import Image

from super_gradients.training.samples import DetectionSample, SegmentationSample, PoseEstimationSample
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh


class SampleType(Enum):
    DETECTION = "DETECTION"
    SEGMENTATION = "SEGMENTATION"
    POSE_ESTIMATION = "POSE_ESTIMATION"
    IMAGE_ONLY = "IMAGE_ONLY"


class TransformsPipelineAdaptorBase(ABC):
    def __init__(self, composed_transforms: Callable):
        self.composed_transforms = composed_transforms

    @abstractmethod
    def __call__(self, sample, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def prep_for_transforms(self, sample):
        raise NotImplementedError

    @abstractmethod
    def post_transforms_processing(self, sample):
        raise NotImplementedError


class AlbumentationsAdaptor(TransformsPipelineAdaptorBase):
    def __init__(self, composed_transforms: Callable):
        super(AlbumentationsAdaptor, self).__init__(composed_transforms)
        self.sample_type = None

    def __call__(self, sample, *args, **kwargs):
        if isinstance(sample, DetectionSample):
            self.sample_type = SampleType.DETECTION
        elif isinstance(sample, SegmentationSample):
            self.sample_type = SampleType.SEGMENTATION
        elif isinstance(sample, PoseEstimationSample):
            self.sample_type = SampleType.POSE_ESTIMATION

            from albumentations.augmentations import HorizontalFlip

            if any(isinstance(t, HorizontalFlip) for t in self.composed_transforms):
                raise TypeError(
                    "`HorizontalFlip` from Albumentation is not supported. "
                    "Please use the `KeypointsRandomHorizontalFlip` from SuperGradients instead.\n"
                    "Note: You should set it like other SuperGradients transforms, and not like other Albumentations."
                )

        else:
            self.sample_type = SampleType.IMAGE_ONLY

        sample = self.prep_for_transforms(sample)
        sample = self.composed_transforms(**sample)
        sample = self.post_transforms_processing(sample)
        return sample

    def apply_to_sample(self, sample):
        return self(sample=sample)

    def prep_for_transforms(self, sample):
        if self.sample_type == SampleType.DETECTION:
            sample = {"image": sample.image, "bboxes": sample.bboxes_xyxy, "labels": sample.labels, "is_crowd": sample.is_crowd}
        elif self.sample_type == SampleType.SEGMENTATION:
            sample = {"image": np.array(sample.image), "mask": np.array(sample.mask)}
        elif self.sample_type == SampleType.POSE_ESTIMATION:

            bboxes_xyxy = xywh_to_xyxy(bboxes=np.array(sample.bboxes_xywh), image_shape=sample.image.shape)

            sample = {
                "image": sample.image,
                "bboxes": bboxes_xyxy,
                "labels": np.zeros(sample.bboxes_xywh.shape[0]),  # Dummy value, this is required for Albumentation. Here, all classes are the same.
                "mask": np.array(sample.mask),
                "is_crowd": sample.is_crowd,
                "keypoints": sample.joints,
            }
        else:
            sample = {"image": np.array(sample)}
        return sample

    def post_transforms_processing(self, sample):
        if self.sample_type == SampleType.DETECTION:
            if len(sample["bboxes"]) == 0:
                sample["bboxes"] = np.zeros((0, 4))
            if len(sample["labels"]) == 0:
                sample["labels"] = np.zeros((0))
            if len(sample["is_crowd"]) == 0:
                sample["is_crowd"] = np.zeros((0))
            sample = DetectionSample(
                image=sample["image"],
                bboxes_xyxy=np.array(sample["bboxes"]),
                labels=np.array(sample["labels"]),
                is_crowd=np.array(sample["is_crowd"]),
                additional_samples=None,
            )
        elif self.sample_type == SampleType.SEGMENTATION:
            sample = SegmentationSample(image=Image.fromarray(sample["image"]), mask=Image.fromarray(sample["mask"]))
        elif self.sample_type == SampleType.POSE_ESTIMATION:

            if len(sample["bboxes"]) == 0:
                sample["bboxes"] = np.zeros((0, 4))
            if len(sample["is_crowd"]) == 0:
                sample["is_crowd"] = np.zeros((0))

            bboxes_xywh = xyxy_to_xywh(bboxes=np.array(sample["bboxes"]), image_shape=sample["image"].shape)

            sample = PoseEstimationSample(
                image=sample["image"],
                mask=np.array(sample["mask"]),
                joints=np.array(sample["keypoints"]),
                areas=None,
                bboxes_xywh=bboxes_xywh,
                is_crowd=np.array(sample["is_crowd"]),
                additional_samples=None,
            )
        else:
            sample = sample["image"]

        return sample
