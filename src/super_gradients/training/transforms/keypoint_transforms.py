import random
from abc import abstractmethod
from typing import Tuple, List, Iterable, Union, Optional, Dict

import cv2
import numpy as np
import torch
from torch import Tensor

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry.registry import register_transform

__all__ = [
    "KeypointsImageNormalize",
    "KeypointsImageStandardize",
    "KeypointsImageToTensor",
    "KeypointsPadIfNeeded",
    "KeypointsLongestMaxSize",
    "KeypointTransform",
    "KeypointsCompose",
    "KeypointsRandomHorizontalFlip",
    "KeypointsRandomAffineTransform",
    "KeypointsRandomVerticalFlip",
]

from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh


@register_transform(Transforms.KeypointTransform)
class KeypointTransform(object):
    """
    Base class for all transforms for keypoints augmentation.
    All transforms subclassing it should implement __call__ method which takes image, mask and keypoints as input and
    returns transformed image, mask and keypoints.
    """

    @abstractmethod
    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply transformation to image, mask and keypoints.

        :param image: Input image of [H,W,3] shape
        :param mask: Numpy array of [H,W] shape, where zero values are considered as ignored mask (not contributing to the loss)
        :param joints: Numpy array of [NumInstances, NumJoints, 3] shape. Last dimension contains (x,y,visibility) for each joint.
        :param areas: (Optional) Numpy array of [N] shape with area of each instance
        :param bboxes: (Optional) Numpy array of [N,4] shape with bounding box of each instance (XYWH)
        :return: (image, mask, joints)
        """
        raise NotImplementedError

    def get_equivalent_preprocessing(self) -> List:
        raise NotImplementedError


class KeypointsCompose(KeypointTransform):
    def __init__(self, transforms: List[KeypointTransform]):
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]
    ) -> Tuple[Union[np.ndarray, Tensor], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        for t in self.transforms:
            image, mask, joints, areas, bboxes = t(image, mask, joints, areas, bboxes)
        return image, mask, joints, areas, bboxes

    def get_equivalent_preprocessing(self) -> List:
        preprocessing = []
        for t in self.transforms:
            preprocessing += t.get_equivalent_preprocessing()
        return preprocessing


@register_transform(Transforms.KeypointsImageToTensor)
class KeypointsImageToTensor(KeypointTransform):
    """
    Convert image from numpy array to tensor and permute axes to [C,H,W].
    """

    def __call__(self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return image, mask, joints, areas, bboxes

    def get_equivalent_preprocessing(self) -> List:
        return [
            {Processings.ImagePermute: {"permutation": (2, 0, 1)}},
        ]


@register_transform(Transforms.KeypointsImageStandardize)
class KeypointsImageStandardize(KeypointTransform):
    """
    Standardize image pixel values with img/max_val

    :param max_val: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        super().__init__()
        self.max_value = max_value

    def __call__(self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        image = (image / self.max_value).astype(np.float32)
        return image, mask, joints, areas, bboxes

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]


@register_transform(Transforms.KeypointsImageNormalize)
class KeypointsImageNormalize(KeypointTransform):
    """
    Normalize image with mean and std.
    """

    def __init__(self, mean, std):
        self.mean = np.array(list(mean)).reshape((1, 1, -1)).astype(np.float32)
        self.std = np.array(list(std)).reshape((1, 1, -1)).astype(np.float32)

    def __call__(self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        image = (image - self.mean) / self.std
        return image, mask, joints, areas, bboxes

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.NormalizeImage: {"mean": self.mean, "std": self.std}}]


@register_transform(Transforms.KeypointsRandomHorizontalFlip)
class KeypointsRandomHorizontalFlip(KeypointTransform):
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
        self.flip_index = flip_index
        self.prob = prob

    def __call__(self, image, mask, joints, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Image shape ({image.shape[:2]}) does not match mask shape ({mask.shape[:2]}).")

        if random.random() < self.prob:
            image = self.apply_to_image(image)
            mask = self.apply_to_image(mask)
            rows, cols = image.shape[:2]

            joints = self.apply_to_keypoints(joints, cols)

            if bboxes is not None:
                bboxes = self.apply_to_bboxes(bboxes, cols)

        return image, mask, joints, areas, bboxes

    def apply_to_image(self, image):
        return np.ascontiguousarray(np.fliplr(image))

    def apply_to_keypoints(self, keypoints, cols):
        keypoints = keypoints.copy()
        keypoints = keypoints[:, self.flip_index]
        keypoints[:, :, 0] = cols - keypoints[:, :, 0] - 1
        return keypoints

    def apply_to_bboxes(self, bboxes, cols):
        bboxes = bboxes.copy()
        bboxes[:, 0] = cols - (bboxes[:, 0] + bboxes[:, 2])
        return bboxes

    def get_equivalent_preprocessing(self) -> List:
        raise RuntimeError("KeypointsRandomHorizontalFlip does not have equivalent preprocessing.")


@register_transform(Transforms.KeypointsRandomVerticalFlip)
class KeypointsRandomVerticalFlip(KeypointTransform):
    """
    Flip image, mask and joints vertically with a given probability.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, mask, joints, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Image shape ({image.shape[:2]}) does not match mask shape ({mask.shape[:2]}).")

        if random.random() < self.prob:
            image = self.apply_to_image(image)
            mask = self.apply_to_image(mask)

            rows, cols = image.shape[:2]
            joints = self.apply_to_keypoints(joints, rows)

            if bboxes is not None:
                bboxes = self.apply_to_bboxes(bboxes, rows)

        return image, mask, joints, areas, bboxes

    def apply_to_image(self, image):
        return np.ascontiguousarray(np.flipud(image))

    def apply_to_keypoints(self, keypoints, rows):
        keypoints = keypoints.copy()
        keypoints[:, :, 1] = rows - keypoints[:, :, 1] - 1
        return keypoints

    def apply_to_bboxes(self, bboxes, rows):
        bboxes = bboxes.copy()
        bboxes[:, 1] = rows - (bboxes[:, 1] + bboxes[:, 3]) - 1
        return bboxes

    def get_equivalent_preprocessing(self) -> List:
        raise RuntimeError("KeypointsRandomHorizontalFlip does not have equivalent preprocessing.")


@register_transform(Transforms.KeypointsLongestMaxSize)
class KeypointsLongestMaxSize(KeypointTransform):
    """
    Resize image, mask and joints to ensure that resulting image does not exceed max_sizes (rows, cols).
    """

    def __init__(self, max_height: int, max_width: int, interpolation: int = cv2.INTER_LINEAR, prob: float = 1.0):
        """

        :param max_sizes: (rows, cols) - Maximum size of the image after resizing
        :param interpolation: Used interpolation method for image
        :param prob: Probability of applying this transform
        """
        self.max_height = max_height
        self.max_width = max_width
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, image, mask, joints, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        if random.random() < self.prob:
            height, width = image.shape[:2]
            scale = min(self.max_height / height, self.max_width / width)
            image = self.apply_to_image(image, scale, cv2.INTER_LINEAR)
            mask = self.apply_to_image(mask, scale, cv2.INTER_LINEAR)

            if image.shape[0] != self.max_height and image.shape[1] != self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={image.shape[:2]})")

            if image.shape[0] > self.max_height or image.shape[1] > self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={image.shape[:2]}")

            joints = self.apply_to_keypoints(joints, scale)
            if bboxes is not None:
                bboxes = self.apply_to_bboxes(bboxes, scale)

            if areas is not None:
                areas = areas * scale

        return image, mask, joints, areas, bboxes

    @classmethod
    def apply_to_image(cls, img, scale, interpolation):
        height, width = img.shape[:2]

        if scale != 1.0:
            new_height, new_width = tuple(int(dim * scale + 0.5) for dim in (height, width))
            img = cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation)
        return img

    @classmethod
    def apply_to_keypoints(cls, keypoints, scale):
        keypoints = keypoints.astype(np.float32, copy=True)
        keypoints[:, :, 0:2] *= scale
        return keypoints

    @classmethod
    def apply_to_bboxes(cls, bboxes, scale):
        return bboxes * scale

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.KeypointsLongestMaxSizeRescale: {"output_shape": (self.max_height, self.max_width)}}]


@register_transform(Transforms.KeypointsPadIfNeeded)
class KeypointsPadIfNeeded(KeypointTransform):
    """
    Pad image and mask to ensure that resulting image size is not less than `output_size` (rows, cols).
    Image and mask padded from right and bottom, thus joints remains unchanged.
    """

    def __init__(self, min_height: int, min_width: int, image_pad_value: int, mask_pad_value: float):
        """

        :param output_size: Desired image size (rows, cols)
        :param image_pad_value: Padding value of image
        :param mask_pad_value: Padding value for mask
        """
        self.min_height = min_height
        self.min_width = min_width
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value

    def __call__(self, image, mask, joints, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        height, width = image.shape[:2]

        pad_bottom = max(0, self.min_height - height)
        pad_right = max(0, self.min_width - width)

        image_pad_value = tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else tuple([self.image_pad_value] * image.shape[-1])
        image = cv2.copyMakeBorder(image, top=0, bottom=pad_bottom, left=0, right=pad_right, value=image_pad_value, borderType=cv2.BORDER_CONSTANT)

        original_dtype = mask.dtype
        mask = cv2.copyMakeBorder(
            mask.astype(np.uint8), top=0, bottom=pad_bottom, left=0, right=pad_right, value=self.mask_pad_value, borderType=cv2.BORDER_CONSTANT
        )
        mask = mask.astype(original_dtype)

        return image, mask, joints, areas, bboxes

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.KeypointsBottomRightPadding: {"output_shape": (self.min_height, self.min_width), "pad_value": self.image_pad_value}}]


@register_transform(Transforms.KeypointsRandomAffineTransform)
class KeypointsRandomAffineTransform(KeypointTransform):
    """
    Apply random affine transform to image, mask and joints.
    """

    def __init__(
        self,
        max_rotation: float,
        min_scale: float,
        max_scale: float,
        max_translate: float,
        image_pad_value: int,
        mask_pad_value: float,
        prob: float = 0.5,
    ):
        """

        :param max_rotation: Max rotation angle in degrees
        :param min_scale: Lower bound for the scale change. For +- 20% size jitter this should be 0.8
        :param max_scale: Lower bound for the scale change. For +- 20% size jitter this should be 1.2
        :param max_translate: Max translation offset in percents of image size
        """
        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_translate = max_translate
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value
        self.prob = prob

    def _get_affine_matrix(self, img, angle, scale, dx, dy):
        """

        :param center: (x,y)
        :param scale:
        :param output_size: (rows, cols)
        :param rot:
        :return:
        """
        height, width = img.shape[:2]
        center = (width / 2 + dx * width, height / 2 + dy * height)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)

        return matrix

    def __call__(self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray, areas: Optional[np.ndarray], bboxes: Optional[np.ndarray]):
        """

        :param image: (np.ndarray) Image of shape [H,W,3]
        :param mask: Single-element array with mask of [H,W] shape.
        :param joints: Single-element array of joints of [Num instances, Num Joints, 3] shape. Semantics of last channel is: x, y, joint index (?)
        :param area: Area each instance occipy: [Num instances, 1]
        :return:
        """

        if random.random() < self.prob:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            scale = random.uniform(self.min_scale, self.max_scale)
            dx = random.uniform(-self.max_translate, self.max_translate)
            dy = random.uniform(-self.max_translate, self.max_translate)

            mat_output = self._get_affine_matrix(image, angle, scale, dx, dy)
            mat_output = mat_output[:2]

            image_pad_value = tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else tuple([self.image_pad_value] * image.shape[-1])

            mask = self.apply_to_image(mask, mat_output, cv2.INTER_NEAREST, self.mask_pad_value, cv2.BORDER_CONSTANT)
            image = self.apply_to_image(image, mat_output, cv2.INTER_LINEAR, image_pad_value, cv2.BORDER_CONSTANT)

            joints = self.apply_to_keypoints(joints, mat_output, image.shape)

            if bboxes is not None:
                bboxes = self.apply_to_bboxes(bboxes, mat_output)

            if areas is not None:
                areas = self.apply_to_areas(areas, mat_output)

        return image, mask, joints, areas, bboxes

    @classmethod
    def apply_to_areas(cls, areas, mat):
        det = np.linalg.det(mat[:2, :2])
        return areas * abs(det)

    @classmethod
    def apply_to_bboxes(cls, bboxes, mat: np.ndarray):
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

        bboxes_xyxy = xywh_to_xyxy(bboxes, image_shape=None)
        bboxes_xyxy = np.array([bbox_shift_scale_rotate(box, mat) for box in bboxes_xyxy])
        return xyxy_to_xywh(bboxes_xyxy, image_shape=None)

    @classmethod
    def apply_to_keypoints(cls, keypoints: np.ndarray, mat: np.ndarray, image_shape):
        keypoints_with_visibility = keypoints.copy()
        keypoints = keypoints_with_visibility[:, :, 0:2]

        shape = keypoints.shape
        keypoints = keypoints.reshape(-1, 2)
        keypoints = np.dot(np.concatenate((keypoints, keypoints[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)

        # Update visibility status of joints that were moved outside visible area
        keypoints_with_visibility[:, :, 0:2] = keypoints
        joints_outside_image = (
            (keypoints[:, :, 0] < 0) | (keypoints[:, :, 0] >= image_shape[1]) | (keypoints[:, :, 1] < 0) | (keypoints[:, :, 1] >= image_shape[0])
        )
        keypoints_with_visibility[joints_outside_image, 2] = 0
        return keypoints_with_visibility

    @classmethod
    def apply_to_image(cls, image, mat, interpolation, padding_value, padding_mode=cv2.BORDER_CONSTANT):
        return cv2.warpAffine(
            image,
            mat,
            dsize=(image.shape[1], image.shape[0]),
            flags=interpolation,
            borderValue=padding_value,
            borderMode=padding_mode,
        )

    def get_equivalent_preprocessing(self) -> List:
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing.")
