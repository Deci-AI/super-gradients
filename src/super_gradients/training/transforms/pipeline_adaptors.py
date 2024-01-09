from enum import Enum
from typing import Callable
from abc import abstractmethod, ABC
import numpy as np
from PIL import Image

from super_gradients.training.samples import DetectionSample, SegmentationSample, PoseEstimationSample, DepthEstimationSample
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh


class SampleType(Enum):
    DETECTION = "DETECTION"
    SEGMENTATION = "SEGMENTATION"
    POSE_ESTIMATION = "POSE_ESTIMATION"
    DEPTH_ESTIMATION = "DEPTH_ESTIMATION"
    IMAGE_ONLY = "IMAGE_ONLY"


class TransformsPipelineAdaptorBase(ABC):
    def __init__(self, composed_transforms: Callable):
        self.composed_transforms = composed_transforms
        self.additional_samples_count = 0  # Does not Support additional samples logic

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
        elif isinstance(sample, DepthEstimationSample):
            self.sample_type = SampleType.DEPTH_ESTIMATION
        elif isinstance(sample, PoseEstimationSample):
            self.sample_type = SampleType.POSE_ESTIMATION

            if self.composed_transforms.to_dict()["transform"].get("keypoint_params") is None:

                example_str = (
                    ""
                    "   > transforms:\n"
                    "   >    - Albumentations:\n"
                    "   >        Compose:\n"
                    "   >            transforms:\n"
                    "   >                - ...:\n"
                    "   >            keypoint_params: # Leave this empty\n"
                )
                raise ValueError(f"`keypoint_params` is required for `PoseEstimationSample`. You can set it like this :\n{example_str}")

            from albumentations.augmentations import HorizontalFlip

            if any(isinstance(t, HorizontalFlip) for t in self.composed_transforms):
                before_str = (
                    ""
                    "   > transforms:\n"
                    "   >    - Albumentations:\n"
                    "   >        Compose:\n"
                    "   >            transforms:\n"
                    "   >                - HorizontalFlip:\n"
                    "   >                    p: 1\n"
                )

                after_str = (
                    ""
                    "   > transforms:\n"
                    "   >    - KeypointsRandomHorizontalFlip:\n"
                    "   >        prob: 1\n"
                    "   >        flip_index: [ 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
                    "   >        # Note: these indexes are COCO-specific. If you're using a different dataset, you will need to change these accordingly.\n"
                    "   >        # The `flip_index` array defines pairs of keypoints to exchange during a horizontal flip. "
                    "This ensures accurate mapping of corresponding keypoints on mirrored body parts after flipping."
                )

                raise TypeError(
                    "`HorizontalFlip` from Albumentation is not supported. "
                    "Please use the `KeypointsRandomHorizontalFlip` from SuperGradients instead.\n"
                    "Note: You should set it like other SuperGradients transforms, and not like other Albumentations.\n\n"
                    "Example:\n\n"
                    f"FROM \n{before_str}\n"
                    f"TO \n{after_str}\n\n"
                )

        else:
            self.sample_type = SampleType.IMAGE_ONLY

        sample = self.prep_for_transforms(sample)
        sample = self.composed_transforms(**sample)  # Apply albumentation compose
        sample = self.post_transforms_processing(sample)
        return sample

    def apply_to_sample(self, sample):
        return self(sample=sample)

    def prep_for_transforms(self, sample):
        if self.sample_type == SampleType.DETECTION:
            sample = {"image": sample.image, "bboxes": sample.bboxes_xyxy, "labels": sample.labels, "is_crowd": sample.is_crowd}
        elif self.sample_type == SampleType.SEGMENTATION:
            sample = {"image": np.array(sample.image), "mask": np.array(sample.mask)}
        elif self.sample_type == SampleType.DEPTH_ESTIMATION:
            sample = {"image": sample.image, "mask": sample.depth_map}
        elif self.sample_type == SampleType.POSE_ESTIMATION:

            bboxes_xyxy = xywh_to_xyxy(bboxes=np.array(sample.bboxes_xywh), image_shape=sample.image.shape)

            sample = {
                "image": sample.image,
                "bboxes": bboxes_xyxy,
                "labels": np.arange(sample.bboxes_xywh.shape[0]),  # Dummy value, this is required for Albumentation. Here, all classes are the same.
                "mask": np.array(sample.mask),
                "is_crowd": sample.is_crowd,
                "keypoints": sample.joints.reshape(sample.joints.shape[0] * sample.joints.shape[1], 3),  # xy
                "n_joints": sample.joints.shape[1],  # Hold
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
        elif self.sample_type == SampleType.DEPTH_ESTIMATION:
            sample = DepthEstimationSample(image=sample["image"], depth_map=sample["mask"])
        elif self.sample_type == SampleType.POSE_ESTIMATION:

            if len(sample["bboxes"]) == 0:
                sample["bboxes"] = np.zeros((0, 4))
            if len(sample["is_crowd"]) == 0:
                sample["is_crowd"] = np.zeros((0))

            bboxes_xywh = xyxy_to_xywh(bboxes=np.array(sample["bboxes"]), image_shape=sample["image"].shape)

            # Update value of keypoints that are not inside the image anymore.
            h, w = sample["image"].shape[0], sample["image"].shape[1]
            keypoints = np.array(
                [keypoint if (0, 0) <= (keypoint[0], keypoint[1]) < (w, h) else (keypoint[0], keypoint[1], 0) for keypoint in sample["keypoints"]]
            )
            keypoints = keypoints.reshape((-1, sample["n_joints"], 3))  # [n_objects, n_joints, 3] with 3: (x, y, visibility)

            # Remove the objects associated with a bbox that was removed.
            keypoints = keypoints[sample["labels"]]

            sample = PoseEstimationSample(
                image=sample["image"],
                mask=np.array(sample["mask"]),
                joints=keypoints,
                areas=None,
                bboxes_xywh=bboxes_xywh,
                is_crowd=np.array(sample["is_crowd"]),
                additional_samples=None,
            )
            sample = sample.sanitize_sample()
        else:
            sample = sample["image"]

        return sample
