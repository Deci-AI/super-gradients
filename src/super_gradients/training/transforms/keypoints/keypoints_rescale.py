import random
import cv2
import numpy as np
from typing import Tuple, List
from super_gradients.common.registry import register_transform
from super_gradients.common.object_names import Transforms, Processings
from super_gradients.training.samples import PoseEstimationSample

from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsRescale)
class KeypointsRescale(AbstractKeypointTransform):
    """
    Resize image, mask and joints to target size without preserving aspect ratio.
    """

    def __init__(self, height: int, width: int, interpolation: int = cv2.INTER_LINEAR, prob: float = 1.0):
        """
        :param height: Target image height
        :param width: Target image width
        :param interpolation: Used interpolation method for image. See cv2.resize for details.
        :param prob: Probability of applying this transform. Default value is 1, meaning that transform is always applied.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.prob = prob

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply transform to sample.
        :param sample: Input sample
        :return:       Output sample
        """
        if random.random() < self.prob:
            height, width = sample.image.shape[:2]
            sy = self.height / height
            sx = self.width / width

            sample.image = self.apply_to_image(sample.image, dsize=(self.width, self.height), interpolation=self.interpolation)
            sample.mask = self.apply_to_image(sample.mask, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)

            sample.joints = self.apply_to_keypoints(sample.joints, sx, sy)
            if sample.bboxes_xywh is not None:
                sample.bboxes_xywh = self.apply_to_bboxes(sample.bboxes_xywh, sx, sy)

            if sample.areas is not None:
                sample.areas = np.multiply(sample.areas, sx * sy, dtype=np.float32)

        return sample

    @classmethod
    def apply_to_image(cls, img, dsize: Tuple[int, int], interpolation: int) -> np.ndarray:
        """
        Resize image to target size.
        :param img:           Input image
        :param dsize:         Target size (width, height)
        :param interpolation: OpenCV interpolation method
        :return:              Resize image
        """
        img = cv2.resize(img, dsize=dsize, interpolation=interpolation)
        return img

    @classmethod
    def apply_to_keypoints(cls, keypoints: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """
        Resize keypoints to target size.
        :param keypoints: [Num Instances, Num Joints, 3] Input keypoints
        :param sx:        Scale factor along the horizontal axis
        :param sy:        Scale factor along the vertical axis
        :return:          [Num Instances, Num Joints, 3] Resized keypoints
        """
        keypoints = keypoints.astype(np.float32, copy=True)
        keypoints[:, :, 0] *= sx
        keypoints[:, :, 1] *= sy
        return keypoints

    @classmethod
    def apply_to_bboxes(cls, bboxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """
        Resize bounding boxes to target size.

        :param bboxes: Input bounding boxes in XYWH format
        :param sx:     Scale factor along the horizontal axis
        :param sy:     Scale factor along the vertical axis
        :return:       Resized bounding boxes in XYWH format
        """
        bboxes = bboxes.astype(np.float32, copy=True)
        bboxes[:, 0::2] *= sx
        bboxes[:, 1::2] *= sy
        return bboxes

    @classmethod
    def apply_to_areas(cls, areas: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """
        Resize areas to target size.
        :param areas: [N] Array of instance areas
        :param sx:    Scale factor along the horizontal axis
        :param sy:    Scale factor along the vertical axis
        :return:      [N] Array of resized instance areas
        """
        return np.multiply(areas, sx * sy, dtype=np.float32)

    def __repr__(self):
        return self.__class__.__name__ + f"(height={self.height}, " f"width={self.width}, " f"interpolation={self.interpolation}, prob={self.prob})"

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.KeypointsRescale: {"output_shape": (self.height, self.width)}}]
