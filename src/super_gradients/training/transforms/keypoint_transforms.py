from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import typing
from abc import abstractmethod
from typing import Tuple

import cv2
import numpy as np
from torchvision.transforms import functional as F


class KeypointTransform(object):
    @abstractmethod
    def __call__(self, image, mask, joints, area, pose_scale_factor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        raise NotImplementedError


class Compose(KeypointTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask, joints, area, pose_scale_factor):
        for t in self.transforms:
            image, mask, joints, area, pose_scale_factor = t(image, mask, joints, area, pose_scale_factor)
        return image, mask, joints, area, pose_scale_factor

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(KeypointTransform):
    def __call__(self, image, mask, joints, area, pose_scale_factor):
        return F.to_tensor(image), mask, joints, area, pose_scale_factor


class Normalize(KeypointTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, joints, area, pose_scale_factor):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, mask, joints, area, pose_scale_factor


class RandomHorizontalFlip(KeypointTransform):
    def __init__(self, flip_index, output_size, prob=0.5):
        self.flip_index = flip_index
        self.prob = prob
        rows, cols = output_size
        self.output_size = [(rows, cols)]

    def __call__(self, image, mask, joints, area, pose_scale_factor):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if random.random() < self.prob:
            image = np.ascontiguousarray(np.fliplr(image))
            # image = image[:, ::-1] - np.zeros_like(image)
            for i, (rows, cols) in enumerate(self.output_size):
                mask[i] = mask[i][:, ::-1] - np.zeros_like(mask[i])
                joints[i] = joints[i][:, self.flip_index]
                joints[i][:, :, 0] = cols - joints[i][:, :, 0] - 1

        return image, mask, joints, area, pose_scale_factor


class RandomVerticalFlip(KeypointTransform):
    def __init__(self, output_size, prob=0.5):
        self.prob = prob
        rows, cols = output_size
        self.output_size = [(rows, cols)]

    def __call__(self, image, mask, joints, area, pose_scale_factor):
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if random.random() < self.prob:
            image = np.ascontiguousarray(np.flipud(image))
            joints = joints.copy()

            for i, (rows, cols) in enumerate(self.output_size):
                mask[i] = mask[i][::-1, :] - np.zeros_like(mask[i])
                joints[i][:, :, 1] = rows - joints[i][:, :, 1] - 1

        return image, mask, joints, area, pose_scale_factor


class LongestMaxSize(KeypointTransform):
    def __init__(self, max_sizes: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR, p: float = 1.0):
        self.max_height, self.max_width = max_sizes
        self.interpolation = interpolation
        self.p = p

    def __call__(self, image, mask, joints, area, pose_scale_factor: float):
        if random.random() < self.p:
            height, width = image.shape[:2]
            scale = min(self.max_height / height, self.max_width / width)

            image = self.rescale_image(image, scale, cv2.INTER_LINEAR)

            if image.shape[0] != self.max_height and image.shape[1] != self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={image.shape[:2]})")

            if image.shape[0] > self.max_height or image.shape[1] > self.max_width:
                raise RuntimeError(f"Image shape is not as expected (scale={scale}, input_shape={height, width}, resized_shape={image.shape[:2]}")

            for i in range(len(mask)):
                mask[i] = self.rescale_image(mask[i], scale, cv2.INTER_LINEAR)

            joints = joints.copy()
            for i in range(len(joints)):
                joints[i][:, :, 0:2] = joints[i][:, :, 0:2] * scale

            area = area * scale * scale
            pose_scale_factor = pose_scale_factor * scale
        return image, mask, joints, area, pose_scale_factor

    @classmethod
    def rescale_image(cls, img, scale, interpolation):
        height, width = img.shape[:2]

        if scale != 1.0:
            new_height, new_width = tuple(int(dim * scale + 0.5) for dim in (height, width))
            img = cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation)
        return img


class PadIfNeeded(KeypointTransform):
    def __init__(self, output_size: Tuple[int, int], image_pad_value: int, mask_pad_value: float):
        self.min_height, self.min_width = output_size
        self.image_pad_value = tuple(image_pad_value) if isinstance(image_pad_value, typing.Iterable) else int(image_pad_value)
        self.mask_pad_value = mask_pad_value

    def __call__(self, image, mask, joints, area, pose_scale_factor):
        height, width = image.shape[:2]

        pad_bottom = max(0, self.min_height - height)
        pad_right = max(0, self.min_width - width)

        image = cv2.copyMakeBorder(image, top=0, bottom=pad_bottom, left=0, right=pad_right, value=self.image_pad_value, borderType=cv2.BORDER_CONSTANT)

        for i in range(len(mask)):
            original_dtype = mask[i].dtype
            mask[i] = cv2.copyMakeBorder(
                mask[i].astype(np.uint8), top=0, bottom=pad_bottom, left=0, right=pad_right, value=self.mask_pad_value, borderType=cv2.BORDER_CONSTANT
            )
            mask[i] = mask[i].astype(original_dtype)

        return image, mask, joints, area, pose_scale_factor


class RandomAffineTransform(KeypointTransform):
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
        p: float = 0.5,
    ):
        """

        :param input_size: (rows, cols)
        :param output_size: (rows, cols)
        :param max_rotation:
        :param min_scale:
        :param max_scale:
        :param scale_type:
        :param max_translate:
        """
        rows, cols = output_size

        self.input_size = input_size
        self.output_size = [(rows, cols)]

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate
        self.image_pad_value = tuple(image_pad_value) if isinstance(image_pad_value, typing.Iterable) else int(image_pad_value)
        self.mask_pad_value = mask_pad_value
        self.p = p

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

    def __call__(self, image, mask, joints, area, pose_scale_factor):
        """

        :param image: (np.ndarray) Image of shape [H,W,3]
        :param mask: Single-element array with mask of [H,W] shape. TODO: Why it is a list of masks?
        :param joints: Single-element array of joints of [Num instances, Num Joints, 3] shape. Semantics of last channel is: x, y, joint index (?)
        :param area: Area each instance occipy: [Num instances, 1]
        :return:
        """
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        if random.random() < self.p:
            joints = joints.copy()

            height, width = image.shape[:2]

            center = np.array((width / 2, height / 2))
            if self.scale_type == "long":
                scale = max(height, width) / 200
                print("###################please modify range")
            elif self.scale_type == "short":
                scale = min(height, width) / 200
            else:
                raise ValueError("Unkonw scale type: {}".format(self.scale_type))
            aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
            scale *= aug_scale
            aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

            if self.max_translate > 0:
                dx = np.random.randint(-self.max_translate * scale, self.max_translate * scale)
                dy = np.random.randint(-self.max_translate * scale, self.max_translate * scale)
                center[0] += dx
                center[1] += dy

            for i, _output_size in enumerate(self.output_size):
                # _output_size in rows, cols
                mat_output, _ = self._get_affine_matrix(center, scale, _output_size, aug_rot)
                mat_output = mat_output[:2]
                mask[i] = cv2.warpAffine(
                    mask[i],
                    mat_output,
                    (_output_size[1], _output_size[0]),
                    borderValue=self.mask_pad_value,
                    borderMode=cv2.BORDER_CONSTANT,
                )

                joints[i][:, :, 0:2] = self._affine_joints(joints[i][:, :, 0:2], mat_output)

            mat_input, final_scale = self._get_affine_matrix(center, scale, self.input_size, aug_rot)
            mat_input = mat_input[:2]
            area = area * final_scale
            image = cv2.warpAffine(image, mat_input, (self.input_size[1], self.input_size[0]), borderValue=self.image_pad_value, borderMode=cv2.BORDER_CONSTANT)
            pose_scale_factor = pose_scale_factor * final_scale

        return image, mask, joints, area, pose_scale_factor
