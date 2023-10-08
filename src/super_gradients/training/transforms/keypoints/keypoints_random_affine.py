import random
from typing import List, Union, Iterable, Tuple

import cv2
import numpy as np

from super_gradients.common.object_names import Transforms
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsRandomAffineTransform)
class KeypointsRandomAffineTransform(AbstractKeypointTransform):
    """
    Apply random affine transform to image, mask and joints.
    """

    def __init__(
        self,
        max_rotation: float,
        min_scale: float,
        max_scale: float,
        max_translate: float,
        image_pad_value: Union[int, float, List[int]],
        mask_pad_value: float,
        interpolation_mode: Union[int, List[int]] = cv2.INTER_LINEAR,
        prob: float = 0.5,
    ):
        """

        :param max_rotation:       Max rotation angle in degrees
        :param min_scale:          Lower bound for the scale change. For +- 20% size jitter this should be 0.8
        :param max_scale:          Lower bound for the scale change. For +- 20% size jitter this should be 1.2
        :param max_translate:      Max translation offset in percents of image size
        :param image_pad_value:    Value to pad the image during affine transform. Can be single scalar or list.
                                   If a list is provided, it should have the same length as the number of channels in the image.
        :param mask_pad_value:     Value to pad the mask during affine transform.
        :param interpolation_mode: A constant integer or list of integers, specifying the interpolation mode to use.
                                   Possible values for interpolation_mode:
                                     cv2.INTER_NEAREST = 0,
                                     cv2.INTER_LINEAR = 1,
                                     cv2.INTER_CUBIC = 2,
                                     cv2.INTER_AREA = 3,
                                     cv2.INTER_LANCZOS4 = 4
                                   To use random interpolation modes on each call, set interpolation_mode = (0,1,2,3,4)
        :param prob:               Probability to apply the transform.
        """
        super().__init__()

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_translate = max_translate
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value
        self.prob = prob
        self.interpolation_mode = tuple(interpolation_mode) if isinstance(interpolation_mode, Iterable) else (interpolation_mode,)

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(max_rotation={self.max_rotation}, "
            f"min_scale={self.min_scale}, "
            f"max_scale={self.max_scale}, "
            f"max_translate={self.max_translate}, "
            f"image_pad_value={self.image_pad_value}, "
            f"mask_pad_value={self.mask_pad_value}, "
            f"prob={self.prob})"
        )

    def _get_affine_matrix(self, img: np.ndarray, angle: float, scale: float, dx: float, dy: float) -> np.ndarray:
        """
        Compute the affine matrix that combines rotation of image around center, scaling and translation
        according to given parameters. Order of operations is: scale, rotate, translate.

        :param angle: Rotation angle in degrees
        :param scale: Scaling factor
        :param dx:    Translation in x direction
        :param dy:    Translation in y direction
        :return:      Affine matrix [2,3]
        """
        height, width = img.shape[:2]
        center = (width / 2 + dx * width, height / 2 + dy * height)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)

        return matrix

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply transformation to given pose estimation sample.
        Since this transformation apply affine transform some keypoints/bboxes may be moved outside the image.
        After applying the transform, visibility status of joints is updated to reflect the new position of joints.
        Bounding boxes are clipped to image borders.
        If sample contains areas, they are scaled according to the applied affine transform.

        :param sample: A pose estimation sample
        :return:       A transformed pose estimation sample
        """

        if random.random() < self.prob:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            scale = random.uniform(self.min_scale, self.max_scale)
            dx = random.uniform(-self.max_translate, self.max_translate)
            dy = random.uniform(-self.max_translate, self.max_translate)
            interpolation = random.choice(self.interpolation_mode)

            mat_output = self._get_affine_matrix(sample.image, angle, scale, dx, dy)
            mat_output = mat_output[:2]

            image_pad_value = (
                tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else tuple([self.image_pad_value] * sample.image.shape[-1])
            )

            sample.image = self.apply_to_image(
                sample.image, mat_output, interpolation=interpolation, padding_value=image_pad_value, padding_mode=cv2.BORDER_CONSTANT
            )
            sample.mask = self.apply_to_image(
                sample.mask, mat_output, interpolation=cv2.INTER_NEAREST, padding_value=self.mask_pad_value, padding_mode=cv2.BORDER_CONSTANT
            )

            sample.joints = self.apply_to_keypoints(sample.joints, mat_output, sample.image.shape[:2])

            if sample.bboxes_xywh is not None:
                sample.bboxes_xywh = self.apply_to_bboxes(sample.bboxes_xywh, mat_output)

            if sample.areas is not None:
                sample.areas = self.apply_to_areas(sample.areas, mat_output)

            sample = sample.sanitize_sample()

        return sample

    @classmethod
    def apply_to_areas(cls, areas: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """
        Apply affine transform to areas.

        :param areas: [N] Single-dimension array of areas
        :param mat:   [2,3] Affine transformation matrix
        :return:      [N] Single-dimension array of areas
        """
        det = np.linalg.det(mat[:2, :2])
        return (areas * abs(det)).astype(areas.dtype)

    @classmethod
    def apply_to_bboxes(cls, bboxes_xywh: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """

        :param bboxes: (N,4) array of bboxes in XYWH format
        :param mat:    [2,3] Affine transformation matrix
        :return:       (N,4) array of bboxes in XYWH format
        """

        def bbox_shift_scale_rotate(bbox, m):
            x_min, y_min, x_max, y_max = bbox[:4]

            x = np.array([x_min, x_max, x_max, x_min])
            y = np.array([y_min, y_min, y_max, y_max])
            ones = np.ones(shape=(len(x)))
            points_ones = np.vstack([x, y, ones]).transpose()

            tr_points = m.dot(points_ones.T).T

            x_min, x_max = min(tr_points[:, 0]), max(tr_points[:, 0])
            y_min, y_max = min(tr_points[:, 1]), max(tr_points[:, 1])

            return np.array([x_min, y_min, x_max, y_max])

        if len(bboxes_xywh) == 0:
            return bboxes_xywh
        bboxes_xyxy = xywh_to_xyxy(bboxes_xywh, image_shape=None)
        bboxes_xyxy = np.array([bbox_shift_scale_rotate(box, mat) for box in bboxes_xyxy])
        return xyxy_to_xywh(bboxes_xyxy, image_shape=None).astype(bboxes_xywh.dtype)

    @classmethod
    def apply_to_keypoints(cls, keypoints: np.ndarray, mat: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Apply affine transform to keypoints.

        :param keypoints:   [N,K,3] array of keypoints in (x,y,visibility) format
        :param mat:         [2,3] Affine transformation matrix
        :param image_shape: Image shape after applying affine transform (height, width).
                            Used to update visibility status of keypoints.
        :return:            [N,K,3] array of keypoints in (x,y,visibility) format
        """
        keypoints_with_visibility = keypoints.copy()
        keypoints = keypoints_with_visibility[:, :, 0:2]

        shape = keypoints.shape
        dtype = keypoints.dtype
        keypoints = keypoints.reshape(-1, 2)
        keypoints = np.dot(np.concatenate((keypoints, keypoints[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)

        # Update visibility status of joints that were moved outside visible area
        image_height, image_width = image_shape[:2]
        outside_left = keypoints[:, :, 0] < 0
        outside_top = keypoints[:, :, 1] < 0
        outside_right = keypoints[:, :, 0] >= image_width
        outside_bottom = keypoints[:, :, 1] >= image_height

        joints_outside_image = outside_left | outside_top | outside_right | outside_bottom

        keypoints_with_visibility[:, :, 0:2] = keypoints
        keypoints_with_visibility[joints_outside_image, 2] = 0
        return keypoints_with_visibility.astype(dtype, copy=False)

    @classmethod
    def apply_to_image(cls, image: np.ndarray, mat: np.ndarray, interpolation: int, padding_value: Union[int, float, Tuple], padding_mode: int) -> np.ndarray:
        """
        Apply affine transform to image.

        :param image:          Input image
        :param mat:            [2,3] Affine transformation matrix
        :param interpolation:  Interpolation mode. See cv2.warpAffine for details.
        :param padding_value:  Value to pad the image during affine transform. See cv2.warpAffine for details.
        :param padding_mode:   Padding mode. See cv2.warpAffine for details.
        :return:               Transformed image of the same shape as input image.
        """
        return cv2.warpAffine(
            image,
            mat,
            dsize=(image.shape[1], image.shape[0]),
            flags=interpolation,
            borderValue=padding_value,
            borderMode=padding_mode,
        )

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
