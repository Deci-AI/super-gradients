import random
from abc import abstractmethod
from typing import Tuple, List, Iterable, Union

import cv2
import numpy as np
from torch import Tensor
from torchvision.transforms import functional as F

__all__ = [
    "KeypointsImageNormalize",
    "KeypointsImageToTensor",
    "KeypointsPadIfNeeded",
    "KeypointsLongestMaxSize",
    "KeypointTransform",
    "KeypointsCompose",
    "KeypointsRandomHorizontalFlip",
    "KeypointsRandomAffineTransform",
    "KeypointsRandomVerticalFlip",
]


class KeypointTransform(object):
    """
    Base class for all transforms for keypoints augmnetation.
    All transforms subclassing it should implement __call__ method which takes image, mask and keypoints as input and
    returns transformed image, mask and keypoints.
    """

    @abstractmethod
    def __call__(self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply transformation to image, mask and keypoints.

        :param image: Input image of [H,W,3] shape
        :param mask: Numpy array of [H,W] shape, where zero values are considered as ignored mask (not contributing to the loss)
        :param joints: Numpy array of [NumInstances, NumJoints, 3] shape. Last dimension contains (x,y,visibility) for each joint.
        :return: (image, mask, joints)
        """
        raise NotImplementedError


class KeypointsCompose(KeypointTransform):
    def __init__(self, transforms: List[KeypointTransform]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray) -> Tuple[Union[np.ndarray, Tensor], np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, mask, joints = t(image, mask, joints)
        return image, mask, joints


class KeypointsImageToTensor(KeypointTransform):
    """
    Convert image from numpy array to tensor and permute axes to [C,H,W].
    This function also divides image by 255.0 to convert it to [0,1] range.
    """

    def __call__(self, image: np.ndarray, mask: np.ndarray, joints: np.ndarray):
        return F.to_tensor(image), mask, joints


class KeypointsImageNormalize(KeypointTransform):
    """
    Normalize image with mean and std. Note this transform should come after KeypointsImageToTensor
    since it operates on torch Tensor and not numpy array.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, mask: np.ndarray, joints: np.ndarray):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints


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

    def __call__(self, image, mask, joints):
        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Image shape ({image.shape[:2]}) does not match mask shape ({mask.shape[:2]}).")

        if random.random() < self.prob:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))
            rows, cols = image.shape[:2]

            joints = joints.copy()
            joints = joints[:, self.flip_index]
            joints[:, :, 0] = cols - joints[:, :, 0] - 1

        return image, mask, joints


class KeypointsRandomVerticalFlip(KeypointTransform):
    """
    Flip image, mask and joints vertically with a given probability.
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, mask, joints):
        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Image shape ({image.shape[:2]}) does not match mask shape ({mask.shape[:2]}).")

        if random.random() < self.prob:
            image = np.ascontiguousarray(np.flipud(image))
            mask = np.ascontiguousarray(np.flipud(mask))

            rows, cols = image.shape[:2]
            joints = joints.copy()
            joints[:, :, 1] = rows - joints[:, :, 1] - 1

        return image, mask, joints


class KeypointsLongestMaxSize(KeypointTransform):
    """
    Resize image, mask and joints to ensure that resulting image does not exceed max_sizes (rows, cols).
    """

    def __init__(self, max_sizes: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR, prob: float = 1.0):
        """

        :param max_sizes: (rows, cols) - Maximum size of the image after resizing
        :param interpolation: Used interpolation method for image
        :param prob: Probability of applying this transform
        """
        self.max_height, self.max_width = max_sizes
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, image, mask, joints: float):
        if random.random() < self.prob:
            height, width = image.shape[:2]
            scale = min(self.max_height / height, self.max_width / width)
            image = self.rescale_image(image, scale, cv2.INTER_LINEAR)

            if image.shape[0] != self.max_height and image.shape[1] != self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={image.shape[:2]})")

            if image.shape[0] > self.max_height or image.shape[1] > self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={image.shape[:2]}")

            mask = self.rescale_image(mask, scale, cv2.INTER_LINEAR)

            joints = joints.copy()
            joints[:, :, 0:2] = joints[:, :, 0:2] * scale

        return image, mask, joints

    @classmethod
    def rescale_image(cls, img, scale, interpolation):
        height, width = img.shape[:2]

        if scale != 1.0:
            new_height, new_width = tuple(int(dim * scale + 0.5) for dim in (height, width))
            img = cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation)
        return img


class KeypointsPadIfNeeded(KeypointTransform):
    """
    Pad image and mask to ensure that resulting image size is not less than `output_size` (rows, cols).
    Image and mask padded from right and bottom, thus joints remains unchanged.
    """

    def __init__(self, output_size: Tuple[int, int], image_pad_value: int, mask_pad_value: float):
        """

        :param output_size: Desired image size (rows, cols)
        :param image_pad_value: Padding value of image
        :param mask_pad_value: Padding value for mask
        """
        self.min_height, self.min_width = output_size
        self.image_pad_value = tuple(image_pad_value) if isinstance(image_pad_value, Iterable) else int(image_pad_value)
        self.mask_pad_value = mask_pad_value

    def __call__(self, image, mask, joints):
        height, width = image.shape[:2]

        pad_bottom = max(0, self.min_height - height)
        pad_right = max(0, self.min_width - width)

        image = cv2.copyMakeBorder(image, top=0, bottom=pad_bottom, left=0, right=pad_right, value=self.image_pad_value, borderType=cv2.BORDER_CONSTANT)

        original_dtype = mask.dtype
        mask = cv2.copyMakeBorder(
            mask.astype(np.uint8), top=0, bottom=pad_bottom, left=0, right=pad_right, value=self.mask_pad_value, borderType=cv2.BORDER_CONSTANT
        )
        mask = mask.astype(original_dtype)

        return image, mask, joints


class KeypointsRandomAffineTransform(KeypointTransform):
    """
    Apply random affine transform to image, mask and joints.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        output_size: Tuple[int, int],
        max_rotation,
        min_scale,
        max_scale,
        scale_type,
        max_translate,
        image_pad_value: int,
        mask_pad_value: float,
        prob: float = 0.5,
    ):
        """

        :param input_size: (rows, cols)
        :param output_size: (rows, cols)
        :param max_rotation: Max rotation angle in degrees
        :param min_scale: Min scale
        :param max_scale: Max scale
        :param scale_type:
        :param max_translate: Max translation offset in pixels
        """
        rows, cols = output_size

        self.input_size = input_size
        self.output_size = (rows, cols)

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate
        self.image_pad_value = tuple(image_pad_value) if isinstance(image_pad_value, Iterable) else int(image_pad_value)
        self.mask_pad_value = mask_pad_value
        self.prob = prob

    def _get_affine_matrix(self, center, scale, output_size: Tuple[int, int], rot=0):
        """

        :param center: (x,y)
        :param scale:
        :param output_size: (rows, cols)
        :param rot:
        :return:
        """
        output_height, output_width = output_size
        center_x, center_y = center

        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(output_size[0]) / h
        t[1, 1] = float(output_size[1]) / h
        t[0, 2] = output_width * (-float(center_x) / h + 0.5)
        t[1, 2] = output_height * (-float(center_y) / h + 0.5)
        t[2, 2] = 1
        scale = t[0, 0] * t[1, 1]
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -output_width / 2
            t_mat[1, 2] = -output_height / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t, scale

    def _affine_joints(self, joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)

    def __call__(self, image, mask, joints):
        """

        :param image: (np.ndarray) Image of shape [H,W,3]
        :param mask: Single-element array with mask of [H,W] shape.
        :param joints: Single-element array of joints of [Num instances, Num Joints, 3] shape. Semantics of last channel is: x, y, joint index (?)
        :param area: Area each instance occipy: [Num instances, 1]
        :return:
        """

        if random.random() < self.prob:
            height, width = image.shape[:2]

            center = np.array((width / 2, height / 2))
            if self.scale_type == "long":
                scale = max(height, width) / 200
            elif self.scale_type == "short":
                scale = min(height, width) / 200
            else:
                raise ValueError("Unknown scale type: {}".format(self.scale_type))
            aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
            scale *= aug_scale
            aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

            if self.max_translate > 0:
                dx = np.random.randint(-self.max_translate * scale, self.max_translate * scale)
                dy = np.random.randint(-self.max_translate * scale, self.max_translate * scale)
                center[0] += dx
                center[1] += dy

            # _output_size in rows, cols
            mat_output, _ = self._get_affine_matrix(center, scale, self.output_size, aug_rot)
            mat_output = mat_output[:2]
            mask = cv2.warpAffine(
                mask,
                mat_output,
                (self.output_size[1], self.output_size[0]),
                borderValue=self.mask_pad_value,
                borderMode=cv2.BORDER_CONSTANT,
            )

            joints = joints.copy()
            joints[:, :, 0:2] = self._affine_joints(joints[:, :, 0:2], mat_output)

            mat_input, final_scale = self._get_affine_matrix(center, scale, self.input_size, aug_rot)
            mat_input = mat_input[:2]
            image = cv2.warpAffine(image, mat_input, (self.input_size[1], self.input_size[0]), borderValue=self.image_pad_value, borderMode=cv2.BORDER_CONSTANT)

        return image, mask, joints
