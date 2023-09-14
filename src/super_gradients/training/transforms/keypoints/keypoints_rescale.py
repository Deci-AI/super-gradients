import random
import cv2
import numpy as np
from typing import Tuple, List
from super_gradients.common.registry import register_transform
from super_gradients.training.transforms.keypoint_transforms import KeypointTransform, PoseEstimationSample
from super_gradients.common.object_names import Transforms, Processings


@register_transform(Transforms.KeypointsRescale)
class KeypointsRescale(KeypointTransform):
    """
    Resize image, mask and joints to target size without preserving aspect ratio.
    """

    def __init__(self, height: int, width: int, interpolation: int = cv2.INTER_LINEAR, prob: float = 1.0):
        """

        :param max_sizes: (rows, cols) - Maximum size of the image after resizing
        :param interpolation: Used interpolation method for image
        :param prob: Probability of applying this transform
        """
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if random.random() < self.prob:
            height, width = sample.image.shape[:2]
            sy = self.height / height
            sx = self.width / width

            sample.image = self.apply_to_image(sample.image, dsize=(self.width, self.height), interpolation=self.interpolation)
            sample.mask = self.apply_to_image(sample.mask, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)

            sample.joints = self.apply_to_keypoints(sample.joints, sx, sy)
            if sample.bboxes is not None:
                sample.bboxes = self.apply_to_bboxes(sample.bboxes, sx, sy)

            if sample.areas is not None:
                sample.areas = np.multiply(sample.areas, sx * sy, dtype=np.float32)
            sample = self.apply_post_transform_sanitization(sample)
        return sample

    @classmethod
    def apply_to_image(cls, img, dsize: Tuple[int, int], interpolation):
        img = cv2.resize(img, dsize=dsize, interpolation=interpolation)
        return img

    @classmethod
    def apply_to_keypoints(cls, keypoints, sx, sy):
        keypoints = keypoints.astype(np.float32, copy=True)
        keypoints[:, :, 0] *= sx
        keypoints[:, :, 1] *= sy
        return keypoints

    @classmethod
    def apply_to_bboxes(cls, bboxes, sx, sy):
        bboxes = bboxes.astype(np.float32, copy=True)
        bboxes[:, 0::2] *= sx
        bboxes[:, 1::2] *= sy
        return bboxes

    def __repr__(self):
        return self.__class__.__name__ + f"(height={self.height}, " f"width={self.width}, " f"interpolation={self.interpolation}, prob={self.prob})"

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.KeypointsRescale: {"output_shape": (self.height, self.width)}}]
