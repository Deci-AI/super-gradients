import collections
import math
import random
from typing import Optional, Union, Tuple, List, Sequence

from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as transforms
import numpy as np
import cv2
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.detection_utils import random_affine, get_mosaic_coordinate, \
    adjust_box_anns, xyxy2cxcywh, _mirror, augment_hsv, rescale_and_pad_to_size

image_resample = Image.BILINEAR
mask_resample = Image.NEAREST

logger = get_logger(__name__)
class SegmentationTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__).replace('{', '(').replace('}', ')')


class ResizeSeg(SegmentationTransform):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        sample["image"] = image.resize((self.w, self.h), image_resample)
        sample["mask"] = mask.resize((self.w, self.h), mask_resample)
        return sample


class RandomFlip(SegmentationTransform):
    """
    Randomly flips the image and mask (synchronously) with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0. <= prob <= 1., f"Probability value must be between 0 and 1, found {prob}"
        self.prob = prob

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            sample["image"] = image
            sample["mask"] = mask

        return sample


class Rescale(SegmentationTransform):
    """
    Rescales the image and mask (synchronously) while preserving aspect ratio.
    The rescaling can be done according to scale_factor, short_size or long_size.
    If more than one argument is given, the rescaling mode is determined by this order: scale_factor, then short_size,
    then long_size.

    Args:
        scale_factor: rescaling is done by multiplying input size by scale_factor:
            out_size = (scale_factor * w, scale_factor * h)
        short_size: rescaling is done by determining the scale factor by the ratio short_size / min(h, w).
        long_size: rescaling is done by determining the scale factor by the ratio long_size / max(h, w).
    """

    def __init__(self,
                 scale_factor: Optional[float] = None,
                 short_size: Optional[int] = None,
                 long_size: Optional[int] = None):
        self.scale_factor = scale_factor
        self.short_size = short_size
        self.long_size = long_size

        self.check_valid_arguments()

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]
        w, h = image.size
        if self.scale_factor is not None:
            scale = self.scale_factor
        elif self.short_size is not None:
            short_size = min(w, h)
            scale = self.short_size / short_size
        else:
            long_size = max(w, h)
            scale = self.long_size / long_size

        out_size = int(scale * w), int(scale * h)

        image = image.resize(out_size, image_resample)
        mask = mask.resize(out_size, mask_resample)

        sample["image"] = image
        sample["mask"] = mask

        return sample

    def check_valid_arguments(self):
        if self.scale_factor is None and self.short_size is None and self.long_size is None:
            raise ValueError("Must assign one rescale argument: scale_factor, short_size or long_size")

        if self.scale_factor is not None and self.scale_factor <= 0:
            raise ValueError(f"Scale factor must be a positive number, found: {self.scale_factor}")
        if self.short_size is not None and self.short_size <= 0:
            raise ValueError(f"Short size must be a positive number, found: {self.short_size}")
        if self.long_size is not None and self.long_size <= 0:
            raise ValueError(f"Long size must be a positive number, found: {self.long_size}")


class RandomRescale:
    """
    Random rescale the image and mask (synchronously) while preserving aspect ratio.
    Scale factor is randomly picked between scales [min, max]
    Args:
        scales: scale range tuple (min, max), if scales is a float range will be defined as (1, scales) if scales > 1,
            otherwise (scales, 1). must be a positive number.
    """

    def __init__(self, scales: Union[float, Tuple, List] = (0.5, 2.0)):
        self.scales = scales

        self.check_valid_arguments()

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]
        w, h = image.size

        scale = random.uniform(self.scales[0], self.scales[1])
        out_size = int(scale * w), int(scale * h)

        image = image.resize(out_size, image_resample)
        mask = mask.resize(out_size, mask_resample)

        sample["image"] = image
        sample["mask"] = mask

        return sample

    def check_valid_arguments(self):
        """
        Check the scale values are valid. if order is wrong, flip the order and return the right scale values.
        """
        if not isinstance(self.scales, collections.abc.Iterable):
            if self.scales <= 1:
                self.scales = (self.scales, 1)
            else:
                self.scales = (1, self.scales)

        if self.scales[0] < 0 or self.scales[1] < 0:
            raise ValueError(f"RandomRescale scale values must be positive numbers, found: {self.scales}")
        if self.scales[0] > self.scales[1]:
            self.scales = (self.scales[1], self.scales[0])
        return self.scales


class RandomRotate(SegmentationTransform):
    """
    Randomly rotates image and mask (synchronously) between 'min_deg' and 'max_deg'.
    """

    def __init__(self, min_deg: float = -10, max_deg: float = 10, fill_mask: int = 0,
                 fill_image: Union[int, Tuple, List] = 0):
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.fill_mask = fill_mask
        # grey color in RGB mode
        self.fill_image = (fill_image, fill_image, fill_image)

        self.check_valid_arguments()

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]

        deg = random.uniform(self.min_deg, self.max_deg)
        image = image.rotate(deg, resample=image_resample, fillcolor=self.fill_image)
        mask = mask.rotate(deg, resample=mask_resample, fillcolor=self.fill_mask)

        sample["image"] = image
        sample["mask"] = mask

        return sample

    def check_valid_arguments(self):
        self.fill_mask, self.fill_image = _validate_fill_values_arguments(self.fill_mask, self.fill_image)


class CropImageAndMask(SegmentationTransform):
    """
    Crops image and mask (synchronously).
    In "center" mode a center crop is performed while, in "random" mode the drop will be positioned around
     random coordinates.
    """

    def __init__(self, crop_size: Union[float, Tuple, List], mode: str):
        """

        :param crop_size: tuple of (width, height) for the final crop size, if is scalar size is a
            square (crop_size, crop_size)
        :param mode: how to choose the center of the crop, 'center' for the center of the input image,
            'random' center the point is chosen randomally
        """

        self.crop_size = crop_size
        self.mode = mode

        self.check_valid_arguments()

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]

        w, h = image.size
        if self.mode == "random":
            x1 = random.randint(0, w - self.crop_size[0])
            y1 = random.randint(0, h - self.crop_size[1])
        else:
            x1 = int(round((w - self.crop_size[0]) / 2.))
            y1 = int(round((h - self.crop_size[1]) / 2.))

        image = image.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        sample["image"] = image
        sample["mask"] = mask

        return sample

    def check_valid_arguments(self):
        if self.mode not in ["center", "random"]:
            raise ValueError(f"Unsupported mode: found: {self.mode}, expected: 'center' or 'random'")

        if not isinstance(self.crop_size, collections.abc.Iterable):
            self.crop_size = (self.crop_size, self.crop_size)
        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
            raise ValueError(f"Crop size must be positive numbers, found: {self.crop_size}")


class RandomGaussianBlur(SegmentationTransform):
    """
    Adds random Gaussian Blur to image with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0. <= prob <= 1., "Probability value must be between 0 and 1"
        self.prob = prob

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]

        if random.random() < self.prob:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        sample["image"] = image
        sample["mask"] = mask

        return sample


class PadShortToCropSize(SegmentationTransform):
    """
    Pads image to 'crop_size'.
    Should be called only after "Rescale" or "RandomRescale" in augmentations pipeline.
    """

    def __init__(self, crop_size: Union[float, Tuple, List], fill_mask: int = 0,
                 fill_image: Union[int, Tuple, List] = 0):
        """
        :param crop_size: tuple of (width, height) for the final crop size, if is scalar size is a
            square (crop_size, crop_size)
        :param fill_mask: value to fill mask labels background.
        :param fill_image: grey value to fill image padded background.
        """
        # CHECK IF CROP SIZE IS A ITERABLE OR SCALAR
        self.crop_size = crop_size
        self.fill_mask = fill_mask
        self.fill_image = tuple(fill_image) if isinstance(fill_image, Sequence) else fill_image

        self.check_valid_arguments()

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]
        w, h = image.size

        # pad images from center symmetrically
        if w < self.crop_size[0] or h < self.crop_size[1]:
            padh = (self.crop_size[1] - h) / 2 if h < self.crop_size[1] else 0
            pad_top, pad_bottom = math.ceil(padh), math.floor(padh)
            padw = (self.crop_size[0] - w) / 2 if w < self.crop_size[0] else 0
            pad_left, pad_right = math.ceil(padw), math.floor(padw)

            image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill_image)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill_mask)

        sample["image"] = image
        sample["mask"] = mask

        return sample

    def check_valid_arguments(self):
        if not isinstance(self.crop_size, collections.abc.Iterable):
            self.crop_size = (self.crop_size, self.crop_size)
        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
            raise ValueError(f"Crop size must be positive numbers, found: {self.crop_size}")

        self.fill_mask, self.fill_image = _validate_fill_values_arguments(self.fill_mask, self.fill_image)


class ColorJitterSeg(transforms.ColorJitter):
    def __call__(self, sample):
        sample["image"] = super(ColorJitterSeg, self).__call__(sample["image"])
        return sample


def _validate_fill_values_arguments(fill_mask: int, fill_image: Union[int, Tuple, List]):
    if not isinstance(fill_image, collections.abc.Iterable):
        # If fill_image is single value, turn to grey color in RGB mode.
        fill_image = (fill_image, fill_image, fill_image)
    elif len(fill_image) != 3:
        raise ValueError(f"fill_image must be an RGB tuple of size equal to 3, found: {fill_image}")
    # assert values are integers
    if not isinstance(fill_mask, int) or not all(isinstance(x, int) for x in fill_image):
        raise ValueError(f"Fill value must be integers,"
                         f" found: fill_image = {fill_image}, fill_mask = {fill_mask}")
    # assert values in range 0-255
    if min(fill_image) < 0 or max(fill_image) > 255 or fill_mask < 0 or fill_mask > 255:
        raise ValueError(f"Fill value must be a value from 0 to 255,"
                         f" found: fill_image = {fill_image}, fill_mask = {fill_mask}")
    return fill_mask, fill_image


class DetectionTransform:
    """
    Detection transform base class.

    Complex transforms that require extra data loading can use the the additional_samples_count attribute in a
     similar fashion to what's been done in COCODetectionDatasetYolox:

    self._load_additional_inputs_for_transform(sample, transform)

    # after the above call, sample["additional_samples"] holds a list of additional inputs and targets.

    sample = transform(sample)



    Attributes:
        additional_samples_count: (int) additional samples to be loaded.
        non_empty_targets: (bool) whether the additianl targets can have empty targets or not.
    """
    def __init__(self, additional_samples_count: int = 0, non_empty_targets: bool = False):
        self.additional_samples_count = additional_samples_count
        self.non_empty_targets = non_empty_targets

    def __call__(self, sample: Union[dict, list]):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__).replace('{', '(').replace('}', ')')


class Mosaic(DetectionTransform):
    """
    Mosaic detection transform
    
    Attributes:
        input_dim: (tuple) input dimension.
        prob: (float) probability of applying mosaic.
        enable_mosaic: (bool) whether to apply mosaic at all (regardless of prob) (default=True).

    """
    def __init__(self, input_dim, prob=1.):
        super(Mosaic, self).__init__(additional_samples_count=3)
        self.prob = prob
        self.input_dim = input_dim
        self.enable_mosaic = True

    def close(self):
        self.additional_samples_count = 0
        self.enable_mosaic = False

    def __call__(self, sample: Union[dict, list]):
        if self.enable_mosaic and random.random() < self.prob:
            mosaic_labels = []
            mosaic_labels_seg = []
            input_h, input_w = self.input_dim[0], self.input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional samples, total of 4
            all_samples = [sample] + sample["additional_samples"]

            for i_mosaic, mosaic_sample in enumerate(all_samples):
                img, _labels, _labels_seg, img_id = mosaic_sample["image"], mosaic_sample["target"], mosaic_sample[
                    "target_seg"], mosaic_sample["id"]
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(i_mosaic, xc, yc, w, h, input_h, input_w)

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                labels_seg = _labels_seg.copy()

                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh

                    labels_seg[:, ::2] = scale * labels_seg[:, ::2] + padw
                    labels_seg[:, 1::2] = scale * labels_seg[:, 1::2] + padh
                mosaic_labels_seg.append(labels_seg)
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                mosaic_labels_seg = np.concatenate(mosaic_labels_seg, 0)
                np.clip(mosaic_labels_seg[:, ::2], 0, 2 * input_w, out=mosaic_labels_seg[:, ::2])
                np.clip(mosaic_labels_seg[:, 1::2], 0, 2 * input_h, out=mosaic_labels_seg[:, 1::2])

            sample = {"image": mosaic_img, "target": mosaic_labels, "target_seg": mosaic_labels_seg,
                      "info": (mosaic_img.shape[1], mosaic_img.shape[0]), "id": sample["id"]}
        return sample


class RandomAffine(DetectionTransform):
    """
    RandomAffine detection transform

    Attributes:
     target_size: (tuple) desired output shape.

     degrees:  (Union[tuple, float]) degrees for random rotation, when float the random values are drawn uniformly
        from (-degrees, degrees)

     translate:  (Union[tuple, float]) translate size (in pixels) for random translation, when float the random values
        are drawn uniformly from (-translate, translate)

     scales: (Union[tuple, float]) values for random rescale, when float the random values are drawn uniformly
        from (0.1-scales, 0.1+scales)

     shear: (Union[tuple, float]) degrees for random shear, when float the random values are drawn uniformly
        from (shear, shear)

    enable: (bool) whether to apply the below transform at all.

    """
    def __init__(self, degrees=10, translate=0.1, scales=0.1, shear=10, target_size=(640, 640)):
        super(RandomAffine, self).__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scales
        self.shear = shear
        self.target_size = target_size
        self.enable = True

    def close(self):
        self.enable = False

    def __call__(self, sample: dict):
        if self.enable:
            img, target = random_affine(
                sample["image"],
                sample["target"],
                sample["target_seg"],
                target_size=self.target_size,
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )
            sample["image"] = img
            sample["target"] = target
        return sample


class Mixup(DetectionTransform):
    """
    Mixup detection transform

    Attributes:
        input_dim: (tuple) input dimension.
        mixup_scale: (tuple) scale range for the additional loaded image for mixup.
        prob: (float) probability of applying mixup.
        enable_mixup: (bool) whether to apply mixup at all (regardless of prob) (default=True).
    """
    def __init__(self, input_dim, mixup_scale, prob=1.):
        super(Mixup, self).__init__(additional_samples_count=1, non_empty_targets=True)
        self.input_dim = input_dim
        self.mixup_scale = mixup_scale
        self.prob = prob
        self.enable_mixup = True

    def close(self):
        self.additional_samples_count = 0
        self.enable_mixup = False

    def __call__(self, sample: dict):
        if self.enable_mixup and random.random() < self.prob:
            origin_img, origin_labels = sample["image"], sample["target"]
            cp_sample = sample["additional_samples"][0]
            img, cp_labels = cp_sample["image"], cp_sample["target"]

            jit_factor = random.uniform(*self.mixup_scale)
            FLIP = random.uniform(0, 1) > 0.5

            if len(img.shape) == 3:
                cp_img = np.ones((self.input_dim[0], self.input_dim[1], 3), dtype=np.uint8) * 114
            else:
                cp_img = np.ones(self.input_dim, dtype=np.uint8) * 114

            cp_scale_ratio = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
                interpolation=cv2.INTER_LINEAR,
            )

            cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
            ] = resized_img

            cp_img = cv2.resize(
                cp_img,
                (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
            )
            cp_scale_ratio *= jit_factor

            if FLIP:
                cp_img = cp_img[:, ::-1, :]

            origin_h, origin_w = cp_img.shape[:2]
            target_h, target_w = origin_img.shape[:2]
            padded_img = np.zeros(
                (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
            )
            padded_img[:origin_h, :origin_w] = cp_img

            x_offset, y_offset = 0, 0
            if padded_img.shape[0] > target_h:
                y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
            if padded_img.shape[1] > target_w:
                x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
            padded_cropped_img = padded_img[
                                 y_offset: y_offset + target_h, x_offset: x_offset + target_w
                                 ]

            cp_bboxes_origin_np = adjust_box_anns(
                cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
            )
            if FLIP:
                cp_bboxes_origin_np[:, 0::2] = (
                        origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
                )
            cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
            cp_bboxes_transformed_np[:, 0::2] = np.clip(
                cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
            )
            cp_bboxes_transformed_np[:, 1::2] = np.clip(
                cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
            )

            cls_labels = cp_labels[:, 4:5].copy()
            box_labels = cp_bboxes_transformed_np
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

            sample["image"], sample["target"] = origin_img.astype(np.uint8), origin_labels
        return sample


class YoloxTrainPreprocessFN(DetectionTransform):
    """
    Preprocessing transform to be applied last of all transforms for training.

    Pads image, converts targets to [cx,cy,w,h] format, flips horizontally, performs hsv augmentation, and pads targets
     to self.max_labels

    Attributes:
        max_labels: (int) size to pad the targets to (default=50).
        flip_prob: (float) probability to apply horizontal flip (default=0.5)
        hsv_prob: (float) probability to apply hsv (default=1.)
        input_dim: (tuple) final input dimension (default=(640,640))
    """
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, input_dim=(640, 640)):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.input_dim = input_dim

    def __call__(self, sample):
        image, targets = sample["image"], sample["target"]
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = rescale_and_pad_to_size(image, self.input_dim)
            sample["image"] = image
            sample["target"] = targets
            return sample

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        image_t, r_ = rescale_and_pad_to_size(image_t, self.input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = rescale_and_pad_to_size(image_o, self.input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
                                                                  : self.max_labels
                                                                  ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        sample["image"], sample["target"] = image_t, padded_labels
        return sample


class YoloxValPreprocessFN(DetectionTransform):
    """
    Preprocessing transform to be applied last of all transforms for validation.

    Rescales and pads to self.input_dim and swaps image channels with self.swap.
    Attributes:
        input_dim: (tuple) final input dimension (default=(640,640))
        swap: image axis's to be rearranged.

    """

    def __init__(self, input_dim, swap=(2, 0, 1)):
        self.swap = swap
        self.input_dim = input_dim

    def __call__(self, sample):
        img, target = sample["image"], sample["target"]
        label = target.copy()
        boxes = label[:, :4]
        img, r = rescale_and_pad_to_size(img, self.input_dim, self.swap)
        boxes = xyxy2cxcywh(boxes)
        boxes *= r
        sample["image"] = img
        sample["target"] = label
        return sample
