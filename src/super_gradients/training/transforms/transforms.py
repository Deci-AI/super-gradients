import collections
import math
import random
import warnings
from numbers import Number
from typing import Optional, Union, Tuple, List, Sequence, Dict

import cv2
import numpy as np
import torch.nn
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as transforms

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry.registry import register_transform
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.data_formats_factory import ConcatenatedTensorFormatFactory
from super_gradients.training.utils.detection_utils import get_mosaic_coordinate, adjust_box_anns, DetectionTargetsFormat
from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormatConverter
from super_gradients.training.datasets.data_formats.formats import filter_on_bboxes, ConcatenatedTensorFormat
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL, LABEL_CXCYWH
from super_gradients.training.transforms.utils import (
    _rescale_and_pad_to_size,
    _rescale_image,
    _rescale_bboxes,
    _get_center_padding_coordinates,
    _pad_image,
    _shift_bboxes,
    _rescale_xyxy_bboxes,
)
from super_gradients.training.utils.utils import ensure_is_tuple_of_two

IMAGE_RESAMPLE_MODE = Image.BILINEAR
MASK_RESAMPLE_MODE = Image.NEAREST

logger = get_logger(__name__)


class SegmentationTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__).replace("{", "(").replace("}", ")")


@register_transform(Transforms.SegResize)
class SegResize(SegmentationTransform):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        sample["image"] = image.resize((self.w, self.h), IMAGE_RESAMPLE_MODE)
        sample["mask"] = mask.resize((self.w, self.h), MASK_RESAMPLE_MODE)
        return sample


@register_transform(Transforms.SegRandomFlip)
class SegRandomFlip(SegmentationTransform):
    """
    Randomly flips the image and mask (synchronously) with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0, f"Probability value must be between 0 and 1, found {prob}"
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask = sample["mask"]
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            sample["image"] = image
            sample["mask"] = mask

        return sample


@register_transform(Transforms.SegRescale)
class SegRescale(SegmentationTransform):
    """
    Rescales the image and mask (synchronously) while preserving aspect ratio.
    The rescaling can be done according to scale_factor, short_size or long_size.
    If more than one argument is given, the rescaling mode is determined by this order: scale_factor, then short_size,
    then long_size.

    :param scale_factor: Rescaling is done by multiplying input size by scale_factor:
            out_size = (scale_factor * w, scale_factor * h)
    :param short_size:  Rescaling is done by determining the scale factor by the ratio short_size / min(h, w).
    :param long_size:   Rescaling is done by determining the scale factor by the ratio long_size / max(h, w).
    """

    def __init__(self, scale_factor: Optional[float] = None, short_size: Optional[int] = None, long_size: Optional[int] = None):
        self.scale_factor = scale_factor
        self.short_size = short_size
        self.long_size = long_size

        self.check_valid_arguments()

    def __call__(self, sample: dict) -> dict:
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

        image = image.resize(out_size, IMAGE_RESAMPLE_MODE)
        mask = mask.resize(out_size, MASK_RESAMPLE_MODE)

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


@register_transform(Transforms.SegRandomRescale)
class SegRandomRescale:
    """
    Random rescale the image and mask (synchronously) while preserving aspect ratio.
    Scale factor is randomly picked between scales [min, max]

    :param scales: Scale range tuple (min, max), if scales is a float range will be defined as (1, scales) if scales > 1,
            otherwise (scales, 1). must be a positive number.
    """

    def __init__(self, scales: Union[float, Tuple, List] = (0.5, 2.0)):
        self.scales = scales

        self.check_valid_arguments()

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask = sample["mask"]
        w, h = image.size

        scale = random.uniform(self.scales[0], self.scales[1])
        out_size = int(scale * w), int(scale * h)

        image = image.resize(out_size, IMAGE_RESAMPLE_MODE)
        mask = mask.resize(out_size, MASK_RESAMPLE_MODE)

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
            raise ValueError(f"SegRandomRescale scale values must be positive numbers, found: {self.scales}")
        if self.scales[0] > self.scales[1]:
            self.scales = (self.scales[1], self.scales[0])
        return self.scales


@register_transform(Transforms.SegRandomRotate)
class SegRandomRotate(SegmentationTransform):
    """
    Randomly rotates image and mask (synchronously) between 'min_deg' and 'max_deg'.
    """

    def __init__(self, min_deg: float = -10, max_deg: float = 10, fill_mask: int = 0, fill_image: Union[int, Tuple, List] = 0):
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.fill_mask = fill_mask
        # grey color in RGB mode
        self.fill_image = (fill_image, fill_image, fill_image)

        self.check_valid_arguments()

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask = sample["mask"]

        deg = random.uniform(self.min_deg, self.max_deg)
        image = image.rotate(deg, resample=IMAGE_RESAMPLE_MODE, fillcolor=self.fill_image)
        mask = mask.rotate(deg, resample=MASK_RESAMPLE_MODE, fillcolor=self.fill_mask)

        sample["image"] = image
        sample["mask"] = mask

        return sample

    def check_valid_arguments(self):
        self.fill_mask, self.fill_image = _validate_fill_values_arguments(self.fill_mask, self.fill_image)


@register_transform(Transforms.SegCropImageAndMask)
class SegCropImageAndMask(SegmentationTransform):
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

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask = sample["mask"]

        w, h = image.size
        if self.mode == "random":
            x1 = random.randint(0, w - self.crop_size[0])
            y1 = random.randint(0, h - self.crop_size[1])
        else:
            x1 = int(round((w - self.crop_size[0]) / 2.0))
            y1 = int(round((h - self.crop_size[1]) / 2.0))

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


@register_transform(Transforms.SegRandomGaussianBlur)
class SegRandomGaussianBlur(SegmentationTransform):
    """
    Adds random Gaussian Blur to image with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0, "Probability value must be between 0 and 1"
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask = sample["mask"]

        if random.random() < self.prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        sample["image"] = image
        sample["mask"] = mask

        return sample


@register_transform(Transforms.SegPadShortToCropSize)
class SegPadShortToCropSize(SegmentationTransform):
    """
    Pads image to 'crop_size'.
    Should be called only after "SegRescale" or "SegRandomRescale" in augmentations pipeline.
    """

    def __init__(self, crop_size: Union[float, Tuple, List], fill_mask: int = 0, fill_image: Union[int, Tuple, List] = 0):
        """
        :param crop_size:   Tuple of (width, height) for the final crop size, if is scalar size is a square (crop_size, crop_size)
        :param fill_mask:   Value to fill mask labels background.
        :param fill_image:  Grey value to fill image padded background.
        """
        # CHECK IF CROP SIZE IS A ITERABLE OR SCALAR
        self.crop_size = crop_size
        self.fill_mask = fill_mask
        self.fill_image = tuple(fill_image) if isinstance(fill_image, Sequence) else fill_image

        self.check_valid_arguments()

    def __call__(self, sample: dict) -> dict:
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


@register_transform(Transforms.SegPadToDivisible)
class SegPadToDivisible(SegmentationTransform):
    def __init__(self, divisible_value: int, fill_mask: int = 0, fill_image: Union[int, Tuple, List] = 0) -> None:
        super().__init__()
        self.divisible_value = divisible_value
        self.fill_mask = fill_mask
        self.fill_image = fill_image
        self.fill_image = tuple(fill_image) if isinstance(fill_image, Sequence) else fill_image

        self.check_valid_arguments()

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        mask = sample["mask"]
        w, h = image.size

        padded_w = int(math.ceil(w / self.divisible_value) * self.divisible_value)
        padded_h = int(math.ceil(h / self.divisible_value) * self.divisible_value)

        if w != padded_w or h != padded_h:
            padh = padded_h - h
            padw = padded_w - w

            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=self.fill_image)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill_mask)

        sample["image"] = image
        sample["mask"] = mask

        return sample

    def check_valid_arguments(self):
        self.fill_mask, self.fill_image = _validate_fill_values_arguments(self.fill_mask, self.fill_image)


@register_transform(Transforms.SegColorJitter)
class SegColorJitter(transforms.ColorJitter):
    def __call__(self, sample):
        sample["image"] = super(SegColorJitter, self).__call__(sample["image"])
        return sample


def _validate_fill_values_arguments(fill_mask: int, fill_image: Union[int, Tuple, List]):
    if not isinstance(fill_image, collections.abc.Iterable):
        # If fill_image is single value, turn to grey color in RGB mode.
        fill_image = (fill_image, fill_image, fill_image)
    elif len(fill_image) != 3:
        raise ValueError(f"fill_image must be an RGB tuple of size equal to 3, found: {fill_image}")
    # assert values are integers
    if not isinstance(fill_mask, int) or not all(isinstance(x, int) for x in fill_image):
        raise ValueError(f"Fill value must be integers," f" found: fill_image = {fill_image}, fill_mask = {fill_mask}")
    # assert values in range 0-255
    if min(fill_image) < 0 or max(fill_image) > 255 or fill_mask < 0 or fill_mask > 255:
        raise ValueError(f"Fill value must be a value from 0 to 255," f" found: fill_image = {fill_image}, fill_mask = {fill_mask}")
    return fill_mask, fill_image


class DetectionTransform:
    """
    Detection transform base class.

    Complex transforms that require extra data loading can use the the additional_samples_count attribute in a
     similar fashion to what's been done in COCODetectionDataset:

    self._load_additional_inputs_for_transform(sample, transform)

    # after the above call, sample["additional_samples"] holds a list of additional inputs and targets.

    sample = transform(sample)


    :param additional_samples_count:    Additional samples to be loaded.
    :param non_empty_targets:           Whether the additional targets can have empty targets or not.
    """

    def __init__(self, additional_samples_count: int = 0, non_empty_targets: bool = False):
        self.additional_samples_count = additional_samples_count
        self.non_empty_targets = non_empty_targets

    def __call__(self, sample: Union[dict, list]):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__).replace("{", "(").replace("}", ")")

    def get_equivalent_preprocessing(self) -> List:
        raise NotImplementedError


@register_transform(Transforms.DetectionStandardize)
class DetectionStandardize(DetectionTransform):
    """
    Standardize image pixel values with img/max_val

    :param max_val: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        super().__init__()
        self.max_value = max_value

    def __call__(self, sample: dict) -> dict:
        sample["image"] = (sample["image"] / self.max_value).astype(np.float32)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]


@register_transform(Transforms.DetectionMosaic)
class DetectionMosaic(DetectionTransform):
    """
    DetectionMosaic detection transform

    :param input_dim:       Input dimension.
    :param prob:            Probability of applying mosaic.
    :param enable_mosaic:   Whether to apply mosaic at all (regardless of prob).
    :param border_value:    Value for filling borders after applying transforms.
    """

    def __init__(self, input_dim: Union[int, Tuple[int, int]], prob: float = 1.0, enable_mosaic: bool = True, border_value=114):
        super(DetectionMosaic, self).__init__(additional_samples_count=3)
        self.prob = prob
        self.input_dim = ensure_is_tuple_of_two(input_dim)
        self.enable_mosaic = enable_mosaic
        self.border_value = border_value

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
                img, _labels = mosaic_sample["image"], mosaic_sample["target"]
                _labels_seg = mosaic_sample.get("target_seg")

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1.0 * input_h / h0, 1.0 * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), self.border_value, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(i_mosaic, xc, yc, w, h, input_h, input_w)

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()

                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

                if _labels_seg is not None:
                    labels_seg = _labels_seg.copy()
                    if _labels.size > 0:
                        labels_seg[:, ::2] = scale * labels_seg[:, ::2] + padw
                        labels_seg[:, 1::2] = scale * labels_seg[:, 1::2] + padh
                    mosaic_labels_seg.append(labels_seg)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            if len(mosaic_labels_seg):
                mosaic_labels_seg = np.concatenate(mosaic_labels_seg, 0)
                np.clip(mosaic_labels_seg[:, ::2], 0, 2 * input_w, out=mosaic_labels_seg[:, ::2])
                np.clip(mosaic_labels_seg[:, 1::2], 0, 2 * input_h, out=mosaic_labels_seg[:, 1::2])

            sample["image"] = mosaic_img
            sample["target"] = mosaic_labels
            sample["info"] = (mosaic_img.shape[1], mosaic_img.shape[0])
            if len(mosaic_labels_seg):
                sample["target_seg"] = mosaic_labels_seg

        return sample


@register_transform(Transforms.DetectionRandomAffine)
class DetectionRandomAffine(DetectionTransform):
    """
    DetectionRandomAffine detection transform

     :param degrees:                Degrees for random rotation, when float the random values are drawn uniformly from (-degrees, degrees)
     :param translate:              Translate size (in pixels) for random translation, when float the random values are drawn uniformly from
                                    (center-translate, center+translate)
     :param scales:                 Values for random rescale, when float the random values are drawn uniformly from (1-scales, 1+scales)
     :param shear:                  Degrees for random shear, when float the random values are drawn uniformly from (-shear, shear)
     :param target_size:            Desired output shape.
     :param filter_box_candidates:  Whether to filter out transformed bboxes by edge size, area ratio, and aspect ratio (default=False).
     :param wh_thr:                 Edge size threshold when filter_box_candidates = True.
                                    Bounding oxes with edges smaller than this values will be filtered out.
     :param ar_thr:                 Aspect ratio threshold filter_box_candidates = True.
                                    Bounding boxes with aspect ratio larger than this values will be filtered out.
     :param area_thr:               Threshold for area ratio between original image and the transformed one, when filter_box_candidates = True.
                                    Bounding boxes with such ratio smaller than this value will be filtered out.
     :param border_value:           Value for filling borders after applying transforms.
    """

    def __init__(
        self,
        degrees: Union[tuple, float] = 10,
        translate: Union[tuple, float] = 0.1,
        scales: Union[tuple, float] = 0.1,
        shear: Union[tuple, float] = 10,
        target_size: Union[int, Tuple[int, int], None] = (640, 640),
        filter_box_candidates: bool = False,
        wh_thr: float = 2,
        ar_thr: float = 20,
        area_thr: float = 0.1,
        border_value: int = 114,
    ):
        super(DetectionRandomAffine, self).__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scales
        self.shear = shear
        self.target_size = ensure_is_tuple_of_two(target_size)
        self.enable = True
        self.filter_box_candidates = filter_box_candidates
        self.wh_thr = wh_thr
        self.ar_thr = ar_thr
        self.area_thr = area_thr
        self.border_value = border_value

    def close(self):
        self.enable = False

    def __call__(self, sample: dict) -> dict:
        if self.enable:
            img, target = random_affine(
                sample["image"],
                sample["target"],
                sample.get("target_seg"),
                target_size=self.target_size or tuple(reversed(sample["image"].shape[:2])),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
                filter_box_candidates=self.filter_box_candidates,
                wh_thr=self.wh_thr,
                area_thr=self.area_thr,
                ar_thr=self.ar_thr,
                border_value=self.border_value,
            )
            sample["image"] = img
            sample["target"] = target
        return sample


@register_transform(Transforms.DetectionMixup)
class DetectionMixup(DetectionTransform):
    """
    Mixup detection transform

    :param input_dim:        Input dimension.
    :param mixup_scale:      Scale range for the additional loaded image for mixup.
    :param prob:             Probability of applying mixup.
    :param enable_mixup:     Whether to apply mixup at all (regardless of prob).
    :param flip_prob:        Probability to apply horizontal flip to the additional sample.
    :param border_value:     Value for filling borders after applying transform.
    """

    def __init__(
        self,
        input_dim: Union[int, Tuple[int, int], None],
        mixup_scale: tuple,
        prob: float = 1.0,
        enable_mixup: bool = True,
        flip_prob: float = 0.5,
        border_value: int = 114,
    ):
        super(DetectionMixup, self).__init__(additional_samples_count=1, non_empty_targets=True)
        self.input_dim = ensure_is_tuple_of_two(input_dim)
        self.mixup_scale = mixup_scale
        self.prob = prob
        self.enable_mixup = enable_mixup
        self.flip_prob = flip_prob
        self.border_value = border_value

    def close(self):
        self.additional_samples_count = 0
        self.enable_mixup = False

    def __call__(self, sample: dict) -> dict:
        if self.enable_mixup and random.random() < self.prob:
            origin_img, origin_labels = sample["image"], sample["target"]
            target_dim = self.input_dim if self.input_dim is not None else sample["image"].shape[:2]

            cp_sample = sample["additional_samples"][0]
            img, cp_labels = cp_sample["image"], cp_sample["target"]
            cp_boxes = cp_labels[:, :4]

            img, cp_boxes = _mirror(img, cp_boxes, self.flip_prob)
            # PLUG IN TARGET THE FLIPPED BOXES
            cp_labels[:, :4] = cp_boxes

            jit_factor = random.uniform(*self.mixup_scale)

            if len(img.shape) == 3:
                cp_img = np.ones((target_dim[0], target_dim[1], 3), dtype=np.uint8) * self.border_value
            else:
                cp_img = np.ones(target_dim, dtype=np.uint8) * self.border_value

            cp_scale_ratio = min(target_dim[0] / img.shape[0], target_dim[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
                interpolation=cv2.INTER_LINEAR,
            )

            cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img

            cp_img = cv2.resize(
                cp_img,
                (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
            )
            cp_scale_ratio *= jit_factor

            origin_h, origin_w = cp_img.shape[:2]
            target_h, target_w = origin_img.shape[:2]

            if len(img.shape) == 3:
                padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), img.shape[2]), dtype=np.uint8)
            else:
                padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w)), dtype=np.uint8)

            padded_img[:origin_h, :origin_w] = cp_img

            x_offset, y_offset = 0, 0
            if padded_img.shape[0] > target_h:
                y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
            if padded_img.shape[1] > target_w:
                x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
            padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

            cp_bboxes_origin_np = adjust_box_anns(cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h)
            cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
            cp_bboxes_transformed_np[:, 0::2] = np.clip(cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w)
            cp_bboxes_transformed_np[:, 1::2] = np.clip(cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h)

            cls_labels = cp_labels[:, 4:5].copy()
            box_labels = cp_bboxes_transformed_np
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

            sample["image"], sample["target"] = origin_img.astype(np.uint8), origin_labels
        return sample


@register_transform(Transforms.DetectionImagePermute)
class DetectionImagePermute(DetectionTransform):
    """
    Permute image dims. Useful for converting image from HWC to CHW format.
    """

    def __init__(self, dims: Tuple[int, int, int] = (2, 0, 1)):
        """

        :param dims: Specify new order of dims. Default value (2, 0, 1) suitable for converting from HWC to CHW format.
        """
        super().__init__()
        self.dims = tuple(dims)

    def __call__(self, sample: Dict[str, np.array]) -> dict:
        sample["image"] = np.ascontiguousarray(sample["image"].transpose(*self.dims))
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.ImagePermute: {"permutation": self.dims}}]


@register_transform(Transforms.DetectionPadToSize)
class DetectionPadToSize(DetectionTransform):
    """
    Preprocessing transform to pad image and bboxes to `input_dim` shape (rows, cols).
    Transform does center padding, so that input image with bboxes located in the center of the produced image.

    Note: This transformation assume that dimensions of input image is equal or less than `output_size`.
    """

    def __init__(self, output_size: Union[int, Tuple[int, int], None], pad_value: int):
        """
        Constructor for DetectionPadToSize transform.

        :param output_size: Output image size (rows, cols)
        :param pad_value: Padding value for image
        """
        super().__init__()
        self.output_size = ensure_is_tuple_of_two(output_size)
        self.pad_value = pad_value

    def __call__(self, sample: dict) -> dict:
        image, targets, crowd_targets = sample["image"], sample["target"], sample.get("crowd_target")
        padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=self.output_size)

        sample["image"] = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=self.pad_value)
        sample["target"] = _shift_bboxes(targets=targets, shift_w=padding_coordinates.left, shift_h=padding_coordinates.top)
        if crowd_targets is not None:
            sample["crowd_target"] = _shift_bboxes(targets=crowd_targets, shift_w=padding_coordinates.left, shift_h=padding_coordinates.top)
        return sample

    def get_equivalent_preprocessing(self) -> List:
        return [{Processings.DetectionCenterPadding: {"output_shape": self.output_size, "pad_value": self.pad_value}}]


@register_transform(Transforms.DetectionPaddedRescale)
class DetectionPaddedRescale(DetectionTransform):
    """
    Preprocessing transform to be applied last of all transforms for validation.

    Image- Rescales and pads to self.input_dim.
    Targets- moves the class label to first index, converts boxes format- xyxy -> cxcywh.

    :param input_dim:   Final input dimension (default=(640,640))
    :param swap:        Image axis's to be rearranged.
    :param pad_value:   Padding value for image.
    """

    def __init__(
        self, input_dim: Union[int, Tuple[int, int], None], swap: Tuple[int, ...] = (2, 0, 1), max_targets: Optional[int] = None, pad_value: int = 114
    ):
        super().__init__()
        _max_targets_deprication(max_targets)
        self.swap = swap
        self.input_dim = ensure_is_tuple_of_two(input_dim)
        self.pad_value = pad_value

    def __call__(self, sample: dict) -> dict:
        img, targets, crowd_targets = sample["image"], sample["target"], sample.get("crowd_target")
        img, r = _rescale_and_pad_to_size(img, self.input_dim, self.swap, self.pad_value)

        sample["image"] = img
        sample["target"] = _rescale_xyxy_bboxes(targets, r)
        if crowd_targets is not None:
            sample["crowd_target"] = _rescale_xyxy_bboxes(crowd_targets, r)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [
            {Processings.DetectionLongestMaxSizeRescale: {"output_shape": self.input_dim}},
            {Processings.DetectionBottomRightPadding: {"output_shape": self.input_dim, "pad_value": self.pad_value}},
            {Processings.ImagePermute: {"permutation": self.swap}},
        ]


@register_transform(Transforms.DetectionHorizontalFlip)
class DetectionHorizontalFlip(DetectionTransform):
    """
    Horizontal Flip for Detection

    :param prob:        Probability of applying horizontal flip
    """

    def __init__(self, prob: float, max_targets: Optional[int] = None):
        super(DetectionHorizontalFlip, self).__init__()
        _max_targets_deprication(max_targets)
        self.prob = prob

    def __call__(self, sample):
        image, targets = sample["image"], sample["target"]
        if len(targets) == 0:
            targets = np.zeros((0, 5), dtype=np.float32)
        boxes = targets[:, :4]
        image, boxes = _mirror(image, boxes, self.prob)
        targets[:, :4] = boxes
        sample["target"] = targets
        sample["image"] = image
        return sample


@register_transform(Transforms.DetectionRescale)
class DetectionRescale(DetectionTransform):
    """
    Resize image and bounding boxes to given image dimensions without preserving aspect ratio

    :param output_shape: (rows, cols)
    """

    def __init__(self, output_shape: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_shape = ensure_is_tuple_of_two(output_shape)

    def __call__(self, sample: dict) -> dict:
        image, targets, crowd_targets = sample["image"], sample["target"], sample.get("crowd_target")

        sy, sx = float(self.output_shape[0]) / float(image.shape[0]), float(self.output_shape[1]) / float(image.shape[1])

        sample["image"] = _rescale_image(image=image, target_shape=self.output_shape)
        sample["target"] = _rescale_bboxes(targets, scale_factors=(sy, sx))
        if crowd_targets is not None:
            sample["crowd_target"] = _rescale_bboxes(crowd_targets, scale_factors=(sy, sx))
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.DetectionRescale: {"output_shape": self.output_shape}}]


@register_transform(Transforms.DetectionRandomRotate90)
class DetectionRandomRotate90(DetectionTransform):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            k = random.randrange(0, 4)

            img, targets, crowd_targets = sample["image"], sample["target"], sample.get("crowd_target")

            sample["image"] = np.ascontiguousarray(np.rot90(img, k))
            sample["target"] = self.rotate_bboxes(targets, k, img.shape[:2])
            if crowd_targets is not None:
                sample["crowd_target"] = self.rotate_bboxes(crowd_targets, k, img.shape[:2])

        return sample

    @classmethod
    def rotate_bboxes(cls, targets, k: int, image_shape):
        if k == 0:
            return targets
        rows, cols = image_shape
        targets = targets.copy()
        targets[:, 0:4] = cls.xyxy_bbox_rot90(targets[:, 0:4], k, rows, cols)
        return targets

    @classmethod
    def xyxy_bbox_rot90(cls, bboxes: np.ndarray, factor: int, rows: int, cols: int):
        """
        Rotates a bounding box by 90 degrees CCW (see np.rot90)

        :param bboxes:  Tensor made of bounding box tuples (x_min, y_min, x_max, y_max).
        :param factor:  Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        :param rows:    Image rows.
        :param cols:    Image cols.

        :return: A bounding box tuple (x_min, y_min, x_max, y_max).

        """
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        if factor == 0:
            bbox = x_min, y_min, x_max, y_max
        elif factor == 1:
            bbox = y_min, cols - x_max, y_max, cols - x_min
        elif factor == 2:
            bbox = cols - x_max, rows - y_max, cols - x_min, rows - y_min
        elif factor == 3:
            bbox = rows - y_max, x_min, rows - y_min, x_max
        else:
            raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
        return np.stack(bbox, axis=1)


@register_transform(Transforms.DetectionRGB2BGR)
class DetectionRGB2BGR(DetectionTransform):
    """
    Detection change Red & Blue channel of the image

    :param prob: Probability to apply the transform.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        if sample["image"].shape[2] < 3:
            raise ValueError("DetectionRGB2BGR transform expects at least 3 channels, got: " + str(sample["image"].shape[2]))

        if random.random() < self.prob:
            sample["image"] = sample["image"][..., ::-1]
        return sample

    def get_equivalent_preprocessing(self) -> List:
        if self.prob < 1:
            raise RuntimeError("Cannot set preprocessing pipeline with randomness. Set prob to 1.")
        return [{Processings.ReverseImageChannels}]


@register_transform(Transforms.DetectionHSV)
class DetectionHSV(DetectionTransform):
    """
    Detection HSV transform.

    :param prob:            Probability to apply the transform.
    :param hgain:           Hue gain.
    :param sgain:           Saturation gain.
    :param vgain:           Value gain.
    :param bgr_channels:    Channel indices of the BGR channels- useful for images with >3 channels, or when BGR channels are in different order.
    """

    def __init__(self, prob: float, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5, bgr_channels=(0, 1, 2)):
        super(DetectionHSV, self).__init__()
        self.prob = prob
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.bgr_channels = bgr_channels
        self._additional_channels_warned = False

    def __call__(self, sample: dict) -> dict:
        if sample["image"].shape[2] < 3:
            raise ValueError("HSV transform expects at least 3 channels, got: " + str(sample["image"].shape[2]))
        if sample["image"].shape[2] > 3 and not self._additional_channels_warned:
            logger.warning(
                "HSV transform received image with "
                + str(sample["image"].shape[2])
                + " channels. HSV transform will only be applied on channels: "
                + str(self.bgr_channels)
                + "."
            )
            self._additional_channels_warned = True
        if random.random() < self.prob:
            augment_hsv(sample["image"], self.hgain, self.sgain, self.vgain, self.bgr_channels)
        return sample


@register_transform(Transforms.DetectionNormalize)
class DetectionNormalize(DetectionTransform):
    """
    Normalize image by subtracting mean and dividing by std.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = np.array(list(mean)).reshape((1, 1, -1)).astype(np.float32)
        self.std = np.array(list(std)).reshape((1, 1, -1)).astype(np.float32)

    def __call__(self, sample: dict) -> dict:
        sample["image"] = (sample["image"] - self.mean) / self.std
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.NormalizeImage: {"mean": self.mean, "std": self.std}}]


@register_transform(Transforms.DetectionTargetsFormatTransform)
class DetectionTargetsFormatTransform(DetectionTransform):
    """
    Detection targets format transform

    Convert targets in input_format to output_format, filter small bboxes and pad targets.

    :param input_dim:          Shape of the images to transform.
    :param input_format:       Format of the input targets. For instance [xmin, ymin, xmax, ymax, cls_id] refers to XYXY_LABEL.
    :param output_format:      Format of the output targets. For instance [xmin, ymin, xmax, ymax, cls_id] refers to XYXY_LABEL
    :param min_bbox_edge_size: bboxes with edge size lower then this values will be removed.
    """

    @resolve_param("input_format", ConcatenatedTensorFormatFactory())
    @resolve_param("output_format", ConcatenatedTensorFormatFactory())
    def __init__(
        self,
        input_dim: Union[int, Tuple[int, int], None] = None,
        input_format: ConcatenatedTensorFormat = XYXY_LABEL,
        output_format: ConcatenatedTensorFormat = LABEL_CXCYWH,
        min_bbox_edge_size: float = 1,
        max_targets: Optional[int] = None,
    ):
        super(DetectionTargetsFormatTransform, self).__init__()
        _max_targets_deprication(max_targets)
        if isinstance(input_format, DetectionTargetsFormat) or isinstance(output_format, DetectionTargetsFormat):
            raise TypeError(
                "DetectionTargetsFormat is not supported for input_format and output_format starting from super_gradients==3.0.7.\n"
                "You can either:\n"
                "\t - use builtin format among super_gradients.training.datasets.data_formats.default_formats.<FORMAT_NAME> (e.g. XYXY_LABEL, CXCY_LABEL, ..)\n"
                "\t - define your custom format using super_gradients.training.datasets.data_formats.formats.ConcatenatedTensorFormat\n"
            )
        self.input_format = input_format
        self.output_format = output_format
        self.min_bbox_edge_size = min_bbox_edge_size
        self.input_dim = None

        if input_dim is not None:
            input_dim = ensure_is_tuple_of_two(input_dim)
            self._setup_input_dim_related_params(input_dim)

    def _setup_input_dim_related_params(self, input_dim: tuple):
        """Setup all the parameters that are related to input_dim."""
        self.input_dim = input_dim
        self.min_bbox_edge_size = self.min_bbox_edge_size / max(input_dim) if self.output_format.bboxes_format.format.normalized else self.min_bbox_edge_size
        self.targets_format_converter = ConcatenatedTensorFormatConverter(
            input_format=self.input_format, output_format=self.output_format, image_shape=input_dim
        )

    def __call__(self, sample: dict) -> dict:

        # if self.input_dim not set yet, it will be set with first batch
        if self.input_dim is None:
            self._setup_input_dim_related_params(input_dim=sample["image"].shape[1:])

        sample["target"] = self.apply_on_targets(sample["target"])
        if "crowd_target" in sample.keys():
            sample["crowd_target"] = self.apply_on_targets(sample["crowd_target"])
        return sample

    def apply_on_targets(self, targets: np.ndarray) -> np.ndarray:
        """Convert targets in input_format to output_format, filter small bboxes and pad targets"""
        targets = self.targets_format_converter(targets)
        targets = self.filter_small_bboxes(targets)
        return np.ascontiguousarray(targets, dtype=np.float32)

    def filter_small_bboxes(self, targets: np.ndarray) -> np.ndarray:
        """Filter bboxes smaller than specified threshold."""

        def _is_big_enough(bboxes: np.ndarray) -> np.ndarray:
            return np.minimum(bboxes[:, 2], bboxes[:, 3]) > self.min_bbox_edge_size

        targets = filter_on_bboxes(fn=_is_big_enough, tensor=targets, tensor_format=self.output_format)
        return targets

    def get_equivalent_preprocessing(self) -> List:
        return []


def get_aug_params(value: Union[tuple, float], center: float = 0) -> float:
    """
    Generates a random value for augmentations as described below

    :param value:       Range of values for generation. Wen tuple-drawn uniformly between (value[0], value[1]), and (center - value, center + value) when float.
    :param center:      Center to subtract when value is float.
    :return:            Generated value
    """
    if isinstance(value, Number):
        return random.uniform(center - float(value), center + float(value))
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                value
            )
        )


def get_affine_matrix(
    input_size: Tuple[int, int],
    target_size: Tuple[int, int],
    degrees: Union[tuple, float] = 10,
    translate: Union[tuple, float] = 0.1,
    scales: Union[tuple, float] = 0.1,
    shear: Union[tuple, float] = 10,
) -> np.ndarray:
    """
    Return a random affine transform matrix.

    :param input_size:      Input shape.
    :param target_size:     Desired output shape.
    :param degrees:         Degrees for random rotation, when float the random values are drawn uniformly from (-degrees, degrees)
    :param translate:       Translate size (in pixels) for random translation, when float the random values are drawn uniformly from (-translate, translate)
    :param scales:          Values for random rescale, when float the random values are drawn uniformly from (1-scales, 1+scales)
    :param shear:           Degrees for random shear, when float the random values are drawn uniformly from (-shear, shear)

    :return: affine_transform_matrix, drawn_scale
    """

    # Center in pixels
    center_m = np.eye(3)
    center = (input_size[0] // 2, input_size[1] // 2)
    center_m[0, 2] = -center[1]
    center_m[1, 2] = -center[0]

    # Rotation and scale
    rotation_m = np.eye(3)
    rotation_m[:2] = cv2.getRotationMatrix2D(angle=get_aug_params(degrees), center=(0, 0), scale=get_aug_params(scales, center=1.0))

    # Shear in degrees
    shear_m = np.eye(3)
    shear_m[0, 1] = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_m[1, 0] = math.tan(get_aug_params(shear) * math.pi / 180)

    # Translation in pixels
    translation_m = np.eye(3)
    translation_m[0, 2] = get_aug_params(translate, center=0.5) * target_size[1]
    translation_m[1, 2] = get_aug_params(translate, center=0.5) * target_size[0]

    return (translation_m @ shear_m @ rotation_m @ center_m)[:2]


def apply_affine_to_bboxes(targets, targets_seg, target_size, M):
    num_gts = len(targets)
    twidth, theight = target_size
    # targets_seg = [B x w x h]
    # if any is_not_nan in axis = 1
    seg_is_present_mask = np.logical_or.reduce(~np.isnan(targets_seg), axis=1)
    num_gts_masks = seg_is_present_mask.sum()
    num_gts_boxes = num_gts - num_gts_masks

    if num_gts_boxes:
        # warp corner points
        corner_points = np.ones((num_gts_boxes * 4, 3))
        # x1y1, x2y2, x1y2, x2y1
        corner_points[:, :2] = targets[~seg_is_present_mask][:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(num_gts_boxes * 4, 2)
        corner_points = corner_points @ M.T  # apply affine transform
        corner_points = corner_points.reshape(num_gts_boxes, 8)

        # create new boxes
        corner_xs = corner_points[:, 0::2]
        corner_ys = corner_points[:, 1::2]
        new_bboxes = np.concatenate((np.min(corner_xs, 1), np.min(corner_ys, 1), np.max(corner_xs, 1), np.max(corner_ys, 1))).reshape(4, -1).T
    else:
        new_bboxes = np.ones((0, 4), dtype=np.float32)

    if num_gts_masks:
        # warp segmentation points
        num_seg_points = targets_seg.shape[1] // 2
        corner_points_seg = np.ones((num_gts_masks * num_seg_points, 3))
        corner_points_seg[:, :2] = targets_seg[seg_is_present_mask].reshape(num_gts_masks * num_seg_points, 2)
        corner_points_seg = corner_points_seg @ M.T
        corner_points_seg = corner_points_seg.reshape(num_gts_masks, num_seg_points * 2)

        # create new boxes
        seg_points_xs = corner_points_seg[:, 0::2]
        seg_points_ys = corner_points_seg[:, 1::2]
        new_tight_bboxes = (
            np.concatenate((np.nanmin(seg_points_xs, 1), np.nanmin(seg_points_ys, 1), np.nanmax(seg_points_xs, 1), np.nanmax(seg_points_ys, 1)))
            .reshape(4, -1)
            .T
        )
    else:
        new_tight_bboxes = np.ones((0, 4), dtype=np.float32)

    targets[~seg_is_present_mask, :4] = new_bboxes
    targets[seg_is_present_mask, :4] = new_tight_bboxes

    # clip boxes
    targets[:, [0, 2]] = targets[:, [0, 2]].clip(0, twidth)
    targets[:, [1, 3]] = targets[:, [1, 3]].clip(0, theight)

    return targets


def random_affine(
    img: np.ndarray,
    targets: np.ndarray = (),
    targets_seg: np.ndarray = None,
    target_size: tuple = (640, 640),
    degrees: Union[float, tuple] = 10,
    translate: Union[float, tuple] = 0.1,
    scales: Union[float, tuple] = 0.1,
    shear: Union[float, tuple] = 10,
    filter_box_candidates: bool = False,
    wh_thr=2,
    ar_thr=20,
    area_thr=0.1,
    border_value=114,
):
    """
    Performs random affine transform to img, targets
    :param img:         Input image of shape [h, w, c]
    :param targets:     Input target
    :param targets_seg: Targets derived from segmentation masks
    :param target_size: Desired output shape
    :param degrees:     Degrees for random rotation, when float the random values are drawn uniformly
                            from (-degrees, degrees).
    :param translate:   Translate size (in pixels) for random translation, when float the random values
                            are drawn uniformly from (-translate, translate)
    :param scales:      Values for random rescale, when float the random values are drawn uniformly
                            from (0.1-scales, 0.1+scales)
    :param shear:       Degrees for random shear, when float the random values are drawn uniformly
                                from (shear, shear)

    :param filter_box_candidates:    whether to filter out transformed bboxes by edge size, area ratio, and aspect ratio.
    :param wh_thr: (float) edge size threshold when filter_box_candidates = True. Bounding oxes with edges smaller
      then this values will be filtered out. (default=2)

    :param ar_thr: (float) aspect ratio threshold filter_box_candidates = True. Bounding boxes with aspect ratio larger
      then this values will be filtered out. (default=20)

    :param area_thr:(float) threshold for area ratio between original image and the transformed one, when when filter_box_candidates = True.
      Bounding boxes with such ratio smaller then this value will be filtered out. (default=0.1)

    :param border_value: value for filling borders after applying transforms (default=114).

    :return:            Image and Target with applied random affine
    """

    targets_seg = np.zeros((targets.shape[0], 0)) if targets_seg is None else targets_seg
    M = get_affine_matrix(img.shape[:2], target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(border_value, border_value, border_value))

    # Transform label coordinates
    if len(targets) > 0:
        targets_orig = targets.copy()
        targets = apply_affine_to_bboxes(targets, targets_seg, target_size, M)
        if filter_box_candidates:
            box_candidates_ids = _filter_box_candidates(targets_orig[:, :4], targets[:, :4], wh_thr=wh_thr, ar_thr=ar_thr, area_thr=area_thr)
            targets = targets[box_candidates_ids]
    return img, targets


def _filter_box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):
    """
    compute candidate boxes
        :param box1:        before augment
        :param box2:        after augment
        :param wh_thr:      wh_thr (pixels)
        :param ar_thr:      aspect_ratio_thr
        :param area_thr:    area_ratio
    :return:
    """
    box1 = box1.T
    box2 = box2.T
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def _mirror(image, boxes, prob=0.5):
    """
    Horizontal flips image and bboxes with probability prob.

    :param image: (np.array) image to be flipped.
    :param boxes: (np.array) bboxes to be modified.
    :param prob: probability to perform flipping.
    :return: flipped_image, flipped_bboxes
    """
    flipped_boxes = boxes.copy()
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        flipped_boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, flipped_boxes


def augment_hsv(img: np.array, hgain: float, sgain: float, vgain: float, bgr_channels=(0, 1, 2)):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img[..., bgr_channels], cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    img[..., bgr_channels] = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)  # no return needed


@register_transform(Transforms.Standardize)
class Standardize(torch.nn.Module):
    """
    Standardize image pixel values.
    :return img/max_val

    attributes:
        max_val: float, value to as described above (default=255)
    """

    def __init__(self, max_val=255.0):
        super(Standardize, self).__init__()
        self.max_val = max_val

    def forward(self, img):
        return img / self.max_val


def _max_targets_deprication(max_targets: Optional[int] = None):
    if max_targets is not None:
        warnings.warn(
            "max_targets is deprecated and will be removed in the future, targets are not padded to the max length anymore. "
            "If you are using collate_fn provided by SG, it is safe to simply drop this argument.",
            DeprecationWarning,
        )
