from enum import Enum
from typing import Callable
from abc import abstractmethod, ABC
import numpy as np
from PIL import Image
from super_gradients.training.samples import DetectionSample, SegmentationSample


class SampleType(Enum):
    DETECTION = "DETECTION"
    SEGMENTATION = "SEGMENTATION"
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

        else:
            sample = sample["image"]

        return sample
