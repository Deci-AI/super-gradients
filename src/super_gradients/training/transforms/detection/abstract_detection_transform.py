import abc
from abc import abstractmethod
from typing import List, Any, Dict

import numpy as np

from super_gradients.training.samples import DetectionSample


class AbstractDetectionTransform(abc.ABC):
    """
    Base class for all transforms for keypoints augmentation.
    All transforms subclassing it should implement __call__ method which takes image, mask and keypoints as input and
    returns transformed image, mask and keypoints.

    :param additional_samples_count: Number of additional samples to generate for each image.
                                    This property is used for mixup & mosaic transforms that needs an extra samples.
    """

    def __init__(self, additional_samples_count: int = 0):
        """
        :param additional_samples_count: (int) number of samples that must be extra samples from dataset. Default value is 0.
        """
        self.additional_samples_count = additional_samples_count

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param sample: Dict with following keys:
                        - image: numpy array of [H,W,C] or [C,H,W] format
                        - target: numpy array of [N,5] shape with bounding box of each instance (XYWH)
                        - crowd_targets: numpy array of [N,5] shape with bounding box of each instance (XYWH)
        """
        image, targets = sample["image"], sample["target"]
        if len(targets) == 0:
            targets = np.zeros((0, 5), dtype=np.float32)
        is_crowd = np.zeros(len(targets), dtype=bool)

        crowd_targets = sample.get("crowd_targets")
        if crowd_targets is not None:
            if len(crowd_targets) == 0:
                crowd_targets = np.zeros((0, 5), dtype=np.float32)

            targets = np.concatenate([targets, crowd_targets], axis=0)
            is_crowd = np.concatenate([is_crowd, np.ones(len(crowd_targets), dtype=bool)], axis=0)

        sample = DetectionSample(
            image=image,
            bboxes_xywh=targets[:, :4],
            labels=targets[:, 4:],
            is_crowd=is_crowd,
            additional_samples=None,
        )

        sample = self.apply_to_sample(sample)

        all_targets = np.concatenate([sample.bboxes_xywh, sample.labels], axis=1)
        is_crowd = sample.is_crowd

        return {
            "image": sample.image,
            "target": all_targets[~is_crowd],
            "crowd_targets": all_targets[is_crowd],
        }

    @abstractmethod
    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        """
        Apply transformation to given pose estimation sample.
        Important note - function call may return new object, may modify it in-place.
        This is implementation dependent and if you need to keep original sample intact it
        is recommended to make a copy of it BEFORE passing it to transform.

        :param sample: Input sample to transform.
        :return:       Modified sample (It can be the same instance as input or a new object).
        """
        raise NotImplementedError

    @abstractmethod
    def get_equivalent_preprocessing(self) -> List:
        raise NotImplementedError
