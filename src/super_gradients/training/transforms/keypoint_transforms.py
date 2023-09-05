import dataclasses
import random
from abc import abstractmethod
from typing import List, Iterable, Optional, Dict, Union

import cv2
import numpy as np
import torch

from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry.registry import register_transform

__all__ = [
    "PoseEstimationSample",
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


@dataclasses.dataclass
class PoseEstimationSample:
    """
    :attr image: Input image in [H,W,C] format
    :attr mask: Target mask in [H,W] format
    :attr joints: Target joints in [NumInstances, NumJoints, 3] format. Last dimension contains (x,y,visibility) for each joint.
    :attr areas: (Optional) Numpy array of [N] shape with area of each instance.
    Note this is not a bbox area, but area of the object itself.
    One may use a heuristic `0.53 * box area` as object area approximation if this is not provided.
    :attr bboxes: (Optional) Numpy array of [N,4] shape with bounding box of each instance (XYWH)
    :attr additional_samples: (Optional) List of additional samples for the same image.
    :attr is_crowd: (Optional) Numpy array of [N] shape with is_crowd flag for each instance
    """

    image: np.ndarray
    mask: np.ndarray
    joints: np.ndarray
    areas: Optional[np.ndarray]
    bboxes: Optional[np.ndarray]
    is_crowd: Optional[np.ndarray]
    additional_samples: List["PoseEstimationSample"] = dataclasses.field(default_factory=list)


@register_transform(Transforms.KeypointTransform)
class KeypointTransform(object):
    """
    Base class for all transforms for keypoints augmentation.
    All transforms subclassing it should implement __call__ method which takes image, mask and keypoints as input and
    returns transformed image, mask and keypoints.

    :attr additional_samples_count: Number of additional samples to generate for each image. This is used for mixup augmentation.
    """

    def __init__(self, additional_samples_count: int = 0):
        self.additional_samples_count = additional_samples_count

    @classmethod
    def compute_area_of_joints_bounding_box(self, joints: np.ndarray) -> np.ndarray:
        """
        Compute area of a bounding box for each instance.
        :param joints:  [Num Instances, Num Joints, 3]
        :return: [Num Instances]
        """
        w = np.max(joints[:, :, 0], axis=-1) - np.min(joints[:, :, 0], axis=-1)
        h = np.max(joints[:, :, 1], axis=-1) - np.min(joints[:, :, 1], axis=-1)
        return w * h

    @classmethod
    def filter_invisible_bboxes(cls, sample: PoseEstimationSample, min_area=1) -> PoseEstimationSample:
        if sample.bboxes is None:
            area = cls.compute_area_of_joints_bounding_box(sample.joints)
        else:
            area = sample.bboxes[..., 2:4].prod(axis=-1)

        keep_mask = area > min_area

        sample.joints = sample.joints[keep_mask]
        sample.is_crowd = sample.is_crowd[keep_mask]
        if sample.bboxes is not None:
            sample.bboxes = sample.bboxes[keep_mask]
        if sample.areas is not None:
            sample.areas = sample.areas[keep_mask]
        return sample

    @classmethod
    def filter_invisible_poses(cls, sample: PoseEstimationSample, min_instance_area=1) -> PoseEstimationSample:

        # Filter instances with all invisible keypoints
        visible_joints_mask = sample.joints[:, :, 2] > 0
        keep_mask = np.sum(visible_joints_mask, axis=-1) > 0

        # Filter instances with too small area
        if min_instance_area > 0:
            if sample.areas is None:
                areas = cls.compute_area_of_joints_bounding_box(sample.joints)
            else:
                areas = sample.areas

            keep_area_mask = areas > min_instance_area
            keep_mask &= keep_area_mask

        sample.joints = sample.joints[keep_mask]
        sample.is_crowd = sample.is_crowd[keep_mask]
        if sample.bboxes is not None:
            sample.bboxes = sample.bboxes[keep_mask]
        if sample.areas is not None:
            sample.areas = sample.areas[keep_mask]
        return sample

    @classmethod
    def apply_post_transform_sanitization(cls, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply post-transform sanitization to joints, keypoints, boxes and areax which includes:
        - Clamping box coordinates to stay within image boundaries
        - Updating visibility status of keypoints is they are outside of image boundaries
        - Updating area if bbox clipping occurs
        """

        if torch.is_tensor(sample.image):
            _, image_height, image_width = sample.image.shape
        else:
            image_height, image_width, _ = sample.image.shape

        # Clamp bboxes to image boundaries
        clamped_boxes = xywh_to_xyxy(sample.bboxes, image_shape=None)
        clamped_boxes[..., [0, 2]] = np.clip(clamped_boxes[..., [0, 2]], 0, image_width - 1)
        clamped_boxes[..., [1, 3]] = np.clip(clamped_boxes[..., [1, 3]], 0, image_height - 1)
        clamped_boxes = xyxy_to_xywh(clamped_boxes, image_shape=None)

        # Update joints visibility status
        outside_image_mask = (
            (sample.joints[:, :, 0] < 0) | (sample.joints[:, :, 1] < 0) | (sample.joints[:, :, 0] >= image_width) | (sample.joints[:, :, 1] >= image_height)
        )
        sample.joints[outside_image_mask, 2] = 0

        # Recompute sample areas if they are present
        if sample.areas is not None:
            area_reduction_factor = clamped_boxes[..., 2:4].prod(axis=-1) / (sample.bboxes[..., 2:4].prod(axis=-1) + 1e-6)
            sample.areas = sample.areas * area_reduction_factor

        sample.bboxes = clamped_boxes

        return cls.filter_invisible_poses(sample)

    @abstractmethod
    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
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
        super().__init__(additional_samples_count=0)
        self.transforms = transforms

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def get_equivalent_preprocessing(self) -> List:
        preprocessing = []
        for t in self.transforms:
            preprocessing += t.get_equivalent_preprocessing()
        return preprocessing

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\t{repr(t)}"
        format_string += "\n)"
        return format_string


@register_transform(Transforms.KeypointsImageToTensor)
class KeypointsImageToTensor(KeypointTransform):
    """
    Convert image from numpy array to tensor and permute axes to [C,H,W].
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = torch.from_numpy(np.transpose(sample.image, (2, 0, 1))).float()
        return sample

    def get_equivalent_preprocessing(self) -> List:
        return [
            {Processings.ImagePermute: {"permutation": (2, 0, 1)}},
        ]

    def __repr__(self):
        return self.__class__.__name__ + "()"


@register_transform(Transforms.KeypointsImageStandardize)
class KeypointsImageStandardize(KeypointTransform):
    """
    Standardize image pixel values with img/max_val

    :param max_val: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        super().__init__()
        self.max_value = max_value

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = (sample.image / self.max_value).astype(np.float32)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]

    def __repr__(self):
        return self.__class__.__name__ + f"(max_value={self.max_value})"


@register_transform(Transforms.KeypointsImageNormalize)
class KeypointsImageNormalize(KeypointTransform):
    """
    Normalize image with mean and std.
    """

    def __init__(self, mean, std):
        super().__init__()

        self.mean = np.array(list(mean)).reshape((1, 1, -1)).astype(np.float32)
        self.std = np.array(list(std)).reshape((1, 1, -1)).astype(np.float32)

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        sample.image = (sample.image - self.mean) / self.std
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"

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
        super().__init__()
        self.flip_index = flip_index
        self.prob = prob

    def __repr__(self):
        return self.__class__.__name__ + f"(flip_index={self.flip_index}, prob={self.prob})"

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
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

    def get_equivalent_preprocessing(self) -> List:
        raise RuntimeError("KeypointsRandomHorizontalFlip does not have equivalent preprocessing.")


@register_transform(Transforms.KeypointsRandomVerticalFlip)
class KeypointsRandomVerticalFlip(KeypointTransform):
    """
    Flip image, mask and joints vertically with a given probability.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if sample.image.shape[:2] != sample.mask.shape[:2]:
            raise RuntimeError(f"Image shape ({sample.image.shape[:2]}) does not match mask shape ({sample.mask.shape[:2]}).")

        if random.random() < self.prob:
            sample.image = self.apply_to_image(sample.image)
            sample.mask = self.apply_to_image(sample.mask)

            rows, cols = sample.image.shape[:2]
            sample.joints = self.apply_to_keypoints(sample.joints, rows)

            if sample.bboxes is not None:
                sample.bboxes = self.apply_to_bboxes(sample.bboxes, rows)

        return sample

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

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob})"


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
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        if random.random() < self.prob:
            height, width = sample.image.shape[:2]
            scale = min(self.max_height / height, self.max_width / width)
            sample.image = self.apply_to_image(sample.image, scale, cv2.INTER_LINEAR)
            sample.mask = self.apply_to_image(sample.mask, scale, cv2.INTER_NEAREST)

            if sample.image.shape[0] != self.max_height and sample.image.shape[1] != self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={sample.image.shape[:2]})")

            if sample.image.shape[0] > self.max_height or sample.image.shape[1] > self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={sample.image.shape[:2]}")

            sample.joints = self.apply_to_keypoints(sample.joints, scale)
            if sample.bboxes is not None:
                sample.bboxes = self.apply_to_bboxes(sample.bboxes, scale)

            if sample.areas is not None:
                sample.areas = np.multiply(sample.areas, scale**2, dtype=np.float32)
            sample = self.apply_post_transform_sanitization(sample)
        return sample

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
        return np.multiply(bboxes, scale, dtype=np.float32)

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(max_height={self.max_height}, "
            f"max_width={self.max_width}, "
            f"interpolation={self.interpolation}, prob={self.prob})"
        )

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.KeypointsLongestMaxSizeRescale: {"output_shape": (self.max_height, self.max_width)}}]


@register_transform(Transforms.KeypointsPadIfNeeded)
class KeypointsPadIfNeeded(KeypointTransform):
    """
    Pad image and mask to ensure that resulting image size is not less than `output_size` (rows, cols).
    Image and mask padded from right and bottom, thus joints remains unchanged.
    """

    def __init__(self, min_height: int, min_width: int, image_pad_value: int, mask_pad_value: float, padding_mode: str):
        """

        :param output_size: Desired image size (rows, cols)
        :param image_pad_value: Padding value of image
        :param mask_pad_value: Padding value for mask
        """
        if padding_mode not in ("bottom_right", "center"):
            raise ValueError(f"Unknown padding mode: {padding_mode}. Supported modes: 'bottom_right', 'center'")
        super().__init__()
        self.min_height = min_height
        self.min_width = min_width
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value
        self.padding_mode = padding_mode

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        height, width = sample.image.shape[:2]
        original_dtype = sample.mask.dtype

        if self.padding_mode == "bottom_right":
            pad_left = 0
            pad_top = 0
            pad_bottom = max(0, self.min_height - height)
            pad_right = max(0, self.min_width - width)
        elif self.padding_mode == "center":
            pad_left = max(0, (self.min_width - width) // 2)
            pad_top = max(0, (self.min_height - height) // 2)
            pad_bottom = max(0, self.min_height - height - pad_top)
            pad_right = max(0, self.min_width - width - pad_left)
        else:
            raise RuntimeError(f"Unknown padding mode: {self.padding_mode}")

        image_pad_value = tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else tuple([self.image_pad_value] * sample.image.shape[-1])
        sample.image = cv2.copyMakeBorder(
            sample.image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right, value=image_pad_value, borderType=cv2.BORDER_CONSTANT
        )

        sample.mask = cv2.copyMakeBorder(
            sample.mask.astype(np.uint8),
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            value=self.mask_pad_value,
            borderType=cv2.BORDER_CONSTANT,
        ).astype(original_dtype)

        sample.joints = self.apply_to_keypoints(sample.joints, pad_left, pad_top)
        if sample.bboxes is not None:
            sample.bboxes = self.apply_to_bboxes(sample.bboxes, pad_left, pad_top)

        return sample

    def apply_to_bboxes(self, bboxes: np.ndarray, pad_left, pad_top):
        bboxes = bboxes.copy()
        bboxes[:, 0] += pad_left
        bboxes[:, 1] += pad_top
        return bboxes

    def apply_to_keypoints(self, keypoints: np.ndarray, pad_left, pad_top):
        keypoints = keypoints.copy()
        keypoints[:, :, 0] += pad_left
        keypoints[:, :, 1] += pad_top
        return keypoints

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(min_height={self.min_height}, "
            f"min_width={self.min_width}, "
            f"image_pad_value={self.image_pad_value}, "
            f"mask_pad_value={self.mask_pad_value}, "
            f"padding_mode={self.padding_mode}, "
            f")"
        )

    def get_equivalent_preprocessing(self) -> List:
        if self.padding_mode == "bottom_right":
            return [{Processings.KeypointsBottomRightPadding: {"output_shape": (self.min_height, self.min_width), "pad_value": self.image_pad_value}}]
        else:
            raise RuntimeError(f"KeypointsPadIfNeeded with padding_mode={self.padding_mode} is not implemented.")


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
        interpolation_mode: Union[int, List[int]] = cv2.INTER_LINEAR,
        prob: float = 0.5,
    ):
        """

        :param max_rotation: Max rotation angle in degrees
        :param min_scale: Lower bound for the scale change. For +- 20% size jitter this should be 0.8
        :param max_scale: Lower bound for the scale change. For +- 20% size jitter this should be 1.2
        :param max_translate: Max translation offset in percents of image size
        :param interpolation_mode: A constant integer or list of integers, specifying the interpolation mode to use.
        Possible values for interpolation_mode:
          cv2.INTER_NEAREST = 0,
          cv2.INTER_LINEAR = 1,
          cv2.INTER_CUBIC = 2,
          cv2.INTER_AREA = 3,
          cv2.INTER_LANCZOS4 = 4
        To use random interpolation modes on each call, set interpolation_mode = (0,1,2,3,4)
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

    def __call__(self, sample: PoseEstimationSample) -> PoseEstimationSample:
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

            mat_output = self._get_affine_matrix(sample.image, angle, scale, dx, dy)
            mat_output = mat_output[:2]

            image_pad_value = (
                tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else tuple([self.image_pad_value] * sample.image.shape[-1])
            )

            image_shape = sample.image.shape

            sample.mask = self.apply_to_image(sample.mask, mat_output, cv2.INTER_NEAREST, self.mask_pad_value, cv2.BORDER_CONSTANT)

            interpolation = random.choice(self.interpolation_mode)
            sample.image = self.apply_to_image(sample.image, mat_output, interpolation, image_pad_value, cv2.BORDER_CONSTANT)

            sample.joints = self.apply_to_keypoints(sample.joints, mat_output, image_shape)

            if sample.bboxes is not None:
                sample.bboxes = self.apply_to_bboxes(sample.bboxes, mat_output)

            if sample.areas is not None:
                sample.areas = self.apply_to_areas(sample.areas, mat_output)

            sample = self.apply_post_transform_sanitization(sample)

        return sample

    @classmethod
    def apply_to_areas(cls, areas: np.ndarray, mat):
        det = np.linalg.det(mat[:2, :2])
        return (areas * abs(det)).astype(areas.dtype)

    @classmethod
    def apply_to_bboxes(cls, bboxes_xywh: np.ndarray, mat: np.ndarray):
        """

        :param bboxes: (N,4) array of bboxes in XYWH format
        :param mat:
        :return:
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
    def apply_to_keypoints(cls, keypoints: np.ndarray, mat: np.ndarray, image_shape):
        keypoints_with_visibility = keypoints.copy()
        keypoints = keypoints_with_visibility[:, :, 0:2]

        shape = keypoints.shape
        dtype = keypoints.dtype
        keypoints = keypoints.reshape(-1, 2)
        keypoints = np.dot(np.concatenate((keypoints, keypoints[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)

        # Update visibility status of joints that were moved outside visible area
        keypoints_with_visibility[:, :, 0:2] = keypoints
        joints_outside_image = (
            (keypoints[:, :, 0] < 0) | (keypoints[:, :, 0] >= image_shape[1]) | (keypoints[:, :, 1] < 0) | (keypoints[:, :, 1] >= image_shape[0])
        )
        keypoints_with_visibility[joints_outside_image, 2] = 0
        return keypoints_with_visibility.astype(dtype, copy=False)

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
