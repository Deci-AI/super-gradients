import random
import numpy as np
from typing import List

from super_gradients.common.object_names import Transforms
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsRandomHorizontalFlip)
class KeypointsRandomHorizontalFlip(AbstractKeypointTransform):
    """
    Flip image, mask and joints horizontally with a given probability.
    """

    def __init__(self, flip_index: List[int], prob: float = 0.5):
        """

        :param flip_index: Indexes of keypoints on the flipped image. When doing left-right flip, left hand becomes right hand.
                           So this array contains order of keypoints on the flipped image. This is dataset specific and depends on
                           how keypoints are defined in dataset.
        :param prob: Probability of flipping
        """
        super().__init__()
        self.flip_index = flip_index
        self.prob = prob

    def __repr__(self):
        return self.__class__.__name__ + f"(flip_index={self.flip_index}, prob={self.prob})"

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if sample.image.shape[:2] != sample.mask.shape[:2]:
            raise RuntimeError(f"Image shape ({sample.image.shape[:2]}) does not match mask shape ({sample.mask.shape[:2]}).")

        if random.random() < self.prob:
            sample.image = self.apply_to_image(sample.image)
            sample.mask = self.apply_to_image(sample.mask)
            rows, cols = sample.image.shape[:2]

            sample.joints = self.apply_to_keypoints(sample.joints, cols)

            if sample.bboxes is not None:
                sample.bboxes = self.apply_to_bboxes(sample.bboxes, cols)

        return sample

    def apply_to_image(self, image):
        return np.ascontiguousarray(np.fliplr(image))

    def apply_to_keypoints(self, keypoints, cols):
        keypoints = keypoints.copy()
        keypoints = keypoints[:, self.flip_index]
        keypoints[:, :, 0] = cols - keypoints[:, :, 0] - 1
        return keypoints

    def apply_to_bboxes(self, bboxes, cols):
        """

        :param bboxes: Input boxes of [N,4] shape in XYWH format
        :param cols: Image width
        :return: Flipped boxes
        """

        bboxes = bboxes.copy()
        bboxes[:, 0] = cols - (bboxes[:, 0] + bboxes[:, 2])
        return bboxes

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
