import math
import random
import warnings
from numbers import Number
from typing import Optional, Union, Tuple, List, Sequence, Dict, Iterable

import cv2
import numpy as np
import torch
import torch.nn
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as _transforms

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.data_formats_factory import ConcatenatedTensorFormatFactory
from super_gradients.common.factories.torch_dtype_factory import TorchDtypeFactory
from super_gradients.common.object_names import Transforms, Processings
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormatConverter
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xyxy_to_xywh
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL, LABEL_CXCYWH
from super_gradients.training.datasets.data_formats.formats import filter_on_bboxes, ConcatenatedTensorFormat
from super_gradients.training.samples import DetectionSample, SegmentationSample
from super_gradients.training.transforms.detection import DetectionPadIfNeeded, AbstractDetectionTransform, LegacyDetectionTransformMixin
from super_gradients.training.transforms.segmentation.abstract_segmentation_transform import AbstractSegmentationTransform
from super_gradients.training.transforms.segmentation.legacy_segmentation_transform_mixin import LegacySegmentationTransformMixin
from super_gradients.training.transforms.utils import (
    _rescale_and_pad_to_size,
    _rescale_image,
    _rescale_bboxes,
    _rescale_xyxy_bboxes,
    _compute_scale_factor,
)
from super_gradients.training.utils.detection_utils import (
    get_mosaic_coordinate,
    adjust_box_anns,
    DetectionTargetsFormat,
    change_bbox_bounds_for_image_size_inplace,
)
from super_gradients.training.utils.utils import ensure_is_tuple_of_two

IMAGE_RESAMPLE_MODE = Image.BILINEAR
MASK_RESAMPLE_MODE = Image.NEAREST

logger = get_logger(__name__)


class SegmentationTransform:
    def __call__(self, *args, **kwargs):
        warnings.warn(
            "Inheriting from class SegmentationTransform is deprecated. "
            "If you have a custom detection transform please change the base class to "
            "AbstractDetectionTransform and implement apply_to_sample() method instead of __call__.",
            DeprecationWarning,
        )
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__).replace("{", "(").replace("}", ")")


@register_transform(Transforms.SegResize)
class SegResize(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)
        image = image.resize((self.w, self.h), IMAGE_RESAMPLE_MODE)
        mask = mask.resize((self.w, self.h), MASK_RESAMPLE_MODE)
        return SegmentationSample(image=image, mask=mask)

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.SegmentationResize: {"output_shape": (self.h, self.w)}}]


@register_transform(Transforms.SegRandomFlip)
class SegRandomFlip(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    """
    Randomly flips the image and mask (synchronously) with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0, f"Probability value must be between 0 and 1, found {prob}"
        self.prob = prob

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            sample = SegmentationSample(image=image, mask=mask)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.SegRescale)
class SegRescale(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
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

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)
        w, h = image.size
        scale = _compute_scale_factor(self.scale_factor, self.short_size, self.long_size, w, h)

        out_size = int(scale * w), int(scale * h)
        image = image.resize(out_size, IMAGE_RESAMPLE_MODE)
        mask = mask.resize(out_size, MASK_RESAMPLE_MODE)

        return SegmentationSample(
            image=image,
            mask=mask,
        )

    def check_valid_arguments(self):
        if self.scale_factor is None and self.short_size is None and self.long_size is None:
            raise ValueError("Must assign one rescale argument: scale_factor, short_size or long_size")

        if self.scale_factor is not None and self.scale_factor <= 0:
            raise ValueError(f"Scale factor must be a positive number, found: {self.scale_factor}")
        if self.short_size is not None and self.short_size <= 0:
            raise ValueError(f"Short size must be a positive number, found: {self.short_size}")
        if self.long_size is not None and self.long_size <= 0:
            raise ValueError(f"Long size must be a positive number, found: {self.long_size}")

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.SegmentationRescale: {"scale_factor": self.scale_factor, "short_size": self.short_size, "long_size": self.long_size}}]


@register_transform(Transforms.SegRandomRescale)
class SegRandomRescale(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    """
    Random rescale the image and mask (synchronously) while preserving aspect ratio.
    Scale factor is randomly picked between scales [min, max]

    :param scales: Scale range tuple (min, max), if scales is a float range will be defined as (1, scales) if scales > 1,
            otherwise (scales, 1). must be a positive number.
    """

    def __init__(self, scales: Union[float, Tuple, List] = (0.5, 2.0)):
        self.scales = scales

        self.check_valid_arguments()

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)
        w, h = image.size

        scale = random.uniform(self.scales[0], self.scales[1])
        out_size = int(scale * w), int(scale * h)
        image = image.resize(out_size, IMAGE_RESAMPLE_MODE)
        mask = mask.resize(out_size, MASK_RESAMPLE_MODE)

        return SegmentationSample(image=image, mask=mask)

    def check_valid_arguments(self):
        """
        Check the scale values are valid. if order is wrong, flip the order and return the right scale values.
        """
        if not isinstance(self.scales, Iterable):
            if self.scales <= 1:
                self.scales = (self.scales, 1)
            else:
                self.scales = (1, self.scales)

        if self.scales[0] < 0 or self.scales[1] < 0:
            raise ValueError(f"SegRandomRescale scale values must be positive numbers, found: {self.scales}")
        if self.scales[0] > self.scales[1]:
            self.scales = (self.scales[1], self.scales[0])
        return self.scales

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.SegRandomRotate)
class SegRandomRotate(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
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

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)

        deg = random.uniform(self.min_deg, self.max_deg)
        image = image.rotate(deg, resample=IMAGE_RESAMPLE_MODE, fillcolor=self.fill_image)
        mask = mask.rotate(deg, resample=MASK_RESAMPLE_MODE, fillcolor=self.fill_mask)

        return SegmentationSample(image=image, mask=mask)

    def check_valid_arguments(self):
        self.fill_mask, self.fill_image = _validate_fill_values_arguments(self.fill_mask, self.fill_image)

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.SegCropImageAndMask)
class SegCropImageAndMask(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
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

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)

        w, h = image.size
        if self.mode == "random":
            x1 = random.randint(0, w - self.crop_size[0])
            y1 = random.randint(0, h - self.crop_size[1])
        else:
            x1 = int(round((w - self.crop_size[0]) / 2.0))
            y1 = int(round((h - self.crop_size[1]) / 2.0))

        image = image.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        return SegmentationSample(image=image, mask=mask)

    def check_valid_arguments(self):
        if self.mode not in ["center", "random"]:
            raise ValueError(f"Unsupported mode: found: {self.mode}, expected: 'center' or 'random'")

        if not isinstance(self.crop_size, Iterable):
            self.crop_size = (self.crop_size, self.crop_size)
        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
            raise ValueError(f"Crop size must be positive numbers, found: {self.crop_size}")

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.SegRandomGaussianBlur)
class SegRandomGaussianBlur(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    """
    Adds random Gaussian Blur to image with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0, "Probability value must be between 0 and 1"
        self.prob = prob

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)

        if random.random() < self.prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return SegmentationSample(image=image, mask=sample.mask)

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.SegPadShortToCropSize)
class SegPadShortToCropSize(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    """
    Pads image to 'crop_size'.
    Should be called only after "SegRescale" or "SegRandomRescale" in augmentations pipeline.
    Please note that if input image size > crop size no change will be made to the image.
    This transform only pads the image and mask into "crop_size" if it's larger than image size
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

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)
        w, h = image.size

        # pad images from center symmetrically
        if w < self.crop_size[0] or h < self.crop_size[1]:
            padh = (self.crop_size[1] - h) / 2 if h < self.crop_size[1] else 0
            pad_top, pad_bottom = math.ceil(padh), math.floor(padh)
            padw = (self.crop_size[0] - w) / 2 if w < self.crop_size[0] else 0
            pad_left, pad_right = math.ceil(padw), math.floor(padw)

            image = ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill_image)
            mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.fill_mask)

        return SegmentationSample(image=image, mask=mask)

    def check_valid_arguments(self):
        if not isinstance(self.crop_size, Iterable):
            self.crop_size = (self.crop_size, self.crop_size)
        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
            raise ValueError(f"Crop size must be positive numbers, found: {self.crop_size}")

        self.fill_mask, self.fill_image = _validate_fill_values_arguments(self.fill_mask, self.fill_image)

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.SegmentationPadShortToCropSize: {"crop_size": self.crop_size, "fill_image": self.fill_image}}]


@register_transform(Transforms.SegPadToDivisible)
class SegPadToDivisible(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    def __init__(self, divisible_value: int, fill_mask: int = 0, fill_image: Union[int, Tuple, List] = 0) -> None:
        super().__init__()
        self.divisible_value = divisible_value
        self.fill_mask = fill_mask
        self.fill_image = fill_image
        self.fill_image = tuple(fill_image) if isinstance(fill_image, Sequence) else fill_image

        self.check_valid_arguments()

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        mask = Image.fromarray(sample.mask)
        w, h = image.size

        padded_w = int(math.ceil(w / self.divisible_value) * self.divisible_value)
        padded_h = int(math.ceil(h / self.divisible_value) * self.divisible_value)

        if w != padded_w or h != padded_h:
            padh = padded_h - h
            padw = padded_w - w

            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=self.fill_image)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill_mask)

        return SegmentationSample(image=image, mask=mask)

    def check_valid_arguments(self):
        self.fill_mask, self.fill_image = _validate_fill_values_arguments(self.fill_mask, self.fill_image)

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.SegmentationPadToDivisible: {"divisible_value": self.divisible_value, "fill_image": self.fill_image}}]


@register_transform(Transforms.SegColorJitter)
class SegColorJitter(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self._color_jitter = _transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        image = Image.fromarray(sample.image)
        image = self._color_jitter(image)
        return SegmentationSample(image=image, mask=sample.mask)

    def get_equivalent_preprocessing(self):
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


def _validate_fill_values_arguments(fill_mask: int, fill_image: Union[int, Tuple, List]):
    if not isinstance(fill_image, Iterable):
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
        warnings.warn(
            "Inheriting from DetectionTransform is deprecated. "
            "If you have a custom detection transform please change the base class to "
            "AbstractDetectionTransform and implement apply_to_sample() method instead of __call__.",
            DeprecationWarning,
        )

    def __call__(self, sample: Union[dict, list]):
        raise NotImplementedError

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        """
        Apply transformation to the input detection sample.
        This method exists here for compatibility reasons to ensure a custom transform that inherits from DetectionSample
        would still work.

        :param sample: Input detection sample.
        :return:       Output detection sample.
        """
        sample_dict = LegacyDetectionTransformMixin.convert_detection_sample_to_dict(sample, include_crowd_target=sample.is_crowd.any())
        sample_dict = self(sample_dict)
        return LegacyDetectionTransformMixin.convert_input_dict_to_detection_sample(sample_dict)

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__).replace("{", "(").replace("}", ")")

    def get_equivalent_preprocessing(self) -> List:
        raise NotImplementedError


@register_transform(Transforms.DetectionStandardize)
class DetectionStandardize(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Standardize image pixel values with img/max_val

    :param max_val: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value: float = 255.0):
        super().__init__()
        self.max_value = float(max_value)

    @classmethod
    def apply_to_image(self, image: np.ndarray, max_value: float) -> np.ndarray:
        return (image / max_value).astype(np.float32)

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        sample.image = self.apply_to_image(sample.image, max_value=self.max_value)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]


@register_transform(Transforms.DetectionMosaic)
class DetectionMosaic(AbstractDetectionTransform, LegacyDetectionTransformMixin):
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

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if self.enable_mosaic and random.random() < self.prob:
            mosaic_labels = []
            mosaic_bboxes = []
            mosaic_iscrowd = []

            input_h, input_w = self.input_dim[0], self.input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional samples, total of 4
            all_samples: List[DetectionSample] = [sample] + sample.additional_samples

            for i_mosaic, sample in enumerate(all_samples):
                img = sample.image

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

                bboxes = sample.bboxes_xyxy * scale + np.array([[padw, padh, padw, padh]], dtype=np.float32)

                mosaic_labels.append(sample.labels)
                mosaic_iscrowd.append(sample.is_crowd)
                mosaic_bboxes.append(bboxes)

            mosaic_iscrowd = np.concatenate(mosaic_iscrowd, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)

            # No need to adjust bboxes for image size since DetectionSample constructor will do this anyway
            sample = DetectionSample(
                image=mosaic_img,
                bboxes_xyxy=mosaic_bboxes,
                labels=mosaic_labels,
                is_crowd=mosaic_iscrowd,
                additional_samples=None,
            )

        return sample

    def get_equivalent_preprocessing(self):
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.DetectionRandomAffine)
class DetectionRandomAffine(AbstractDetectionTransform, LegacyDetectionTransformMixin):
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

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if self.enable:
            crowd_mask = sample.is_crowd > 0
            crowd_targets = np.concatenate([sample.bboxes_xyxy[crowd_mask], sample.labels[crowd_mask, None]], axis=1)
            targets = np.concatenate([sample.bboxes_xyxy[~crowd_mask], sample.labels[~crowd_mask, None]], axis=1)

            img, targets, crowd_targets = random_affine(
                sample.image,
                targets=targets,
                targets_seg=None,
                crowd_targets=crowd_targets,
                target_size=self.target_size or tuple(reversed(sample.image.shape[:2])),
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

            is_crowd = np.array([0] * len(targets) + [1] * len(crowd_targets), dtype=bool)
            bboxes_xyxy = np.concatenate([targets[:, 0:4], crowd_targets[:, 0:4]], axis=0, dtype=sample.bboxes_xyxy.dtype)
            labels = np.concatenate([targets[:, 4], crowd_targets[:, 4]], axis=0, dtype=sample.labels.dtype)

            sample = DetectionSample(
                image=img,
                bboxes_xyxy=bboxes_xyxy,
                labels=labels,
                is_crowd=is_crowd,
                additional_samples=None,
            )
        return sample

    def get_equivalent_preprocessing(self):
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.DetectionMixup)
class DetectionMixup(AbstractDetectionTransform, LegacyDetectionTransformMixin):
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
        super(DetectionMixup, self).__init__(additional_samples_count=1)
        self.input_dim = ensure_is_tuple_of_two(input_dim)
        self.mixup_scale = mixup_scale
        self.prob = prob
        self.enable_mixup = enable_mixup
        self.flip_prob = flip_prob
        self.border_value = border_value
        self.non_empty_targets = True
        self.maybe_flip = DetectionHorizontalFlip(prob=flip_prob)

    def close(self):
        self.additional_samples_count = 0
        self.enable_mixup = False

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if self.enable_mixup and random.random() < self.prob:
            (cp_sample,) = sample.additional_samples
            target_dim = self.input_dim if self.input_dim is not None else sample.image.shape[:2]

            cp_sample = self.maybe_flip.apply_to_sample(cp_sample)

            jit_factor = random.uniform(*self.mixup_scale)

            if len(sample.image.shape) == 3:
                cp_img = np.ones((target_dim[0], target_dim[1], sample.image.shape[2]), dtype=np.uint8) * self.border_value
            else:
                cp_img = np.ones(target_dim, dtype=np.uint8) * self.border_value

            cp_scale_ratio = min(target_dim[0] / cp_sample.image.shape[0], target_dim[1] / cp_sample.image.shape[1])
            resized_img = cv2.resize(
                cp_sample.image,
                (int(cp_sample.image.shape[1] * cp_scale_ratio), int(cp_sample.image.shape[0] * cp_scale_ratio)),
                interpolation=cv2.INTER_LINEAR,
            )

            cp_img[: resized_img.shape[0], : resized_img.shape[1]] = resized_img

            cp_img = cv2.resize(
                cp_img,
                (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
                interpolation=cv2.INTER_LINEAR,
            )
            cp_scale_ratio *= jit_factor

            origin_h, origin_w = cp_img.shape[:2]
            target_h, target_w = sample.image.shape[:2]

            if len(cp_img.shape) == 3:
                padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), cp_img.shape[2]), dtype=np.uint8)
            else:
                padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w)), dtype=np.uint8)

            padded_img[:origin_h, :origin_w] = cp_img

            x_offset, y_offset = 0, 0
            if padded_img.shape[0] > target_h:
                y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
            if padded_img.shape[1] > target_w:
                x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
            padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

            cp_bboxes_origin_np = adjust_box_anns(cp_sample.bboxes_xyxy[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h)
            cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
            cp_bboxes_transformed_np[:, [0, 2]] = cp_bboxes_transformed_np[:, [0, 2]] - x_offset
            cp_bboxes_transformed_np[:, [1, 3]] = cp_bboxes_transformed_np[:, [1, 3]] - y_offset
            cp_bboxes_transformed_np = change_bbox_bounds_for_image_size_inplace(cp_bboxes_transformed_np, (target_h, target_w))

            mixup_boxes = np.concatenate([sample.bboxes_xyxy, cp_bboxes_transformed_np], axis=0)
            mixup_labels = np.concatenate([sample.labels, cp_sample.labels], axis=0)
            mixup_crowds = np.concatenate([sample.is_crowd, cp_sample.is_crowd], axis=0)

            mixup_image = (0.5 * sample.image + 0.5 * padded_cropped_img).astype(sample.image.dtype)
            sample = DetectionSample(
                image=mixup_image,
                bboxes_xyxy=mixup_boxes,
                labels=mixup_labels,
                is_crowd=mixup_crowds,
                additional_samples=None,
            )

        return sample

    def get_equivalent_preprocessing(self):
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.SegToTensor)
class SegToTensor:
    def __init__(self, mask_output_dtype: Optional[torch.dtype] = None, add_mask_dummy_dim: bool = False):
        raise NotImplementedError(
            "SegToTensor has been deprecated. Please remove it from your pipeline and add SegConvertToTensor as a LAST transformation instead ."
        )


@register_transform(Transforms.SegConvertToTensor)
class SegConvertToTensor(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    """
    Converts SegmentationSample images and masks to PyTorch tensors.

    :param mask_output_dtype (Optional[str]): The desired output data type for the mask tensor.
    :param add_mask_dummy_dim (bool): Whether to add a dummy channels dimension to the mask tensor.
    """

    @resolve_param("image_output_dtype", TorchDtypeFactory())
    @resolve_param("mask_output_dtype", TorchDtypeFactory())
    def __init__(self, image_output_dtype=torch.float32, mask_output_dtype: Optional[torch.dtype] = None, add_mask_dummy_dim: bool = False):
        self.image_output_dtype = image_output_dtype
        self.mask_output_dtype = mask_output_dtype
        self.add_mask_dummy_dim = add_mask_dummy_dim

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        sample.image = torch.from_numpy(sample.image).permute(2, 0, 1).to(self.image_output_dtype)
        sample.mask = torch.from_numpy(np.array(sample.mask))

        # Convert mask to torch tensor with specified dtype
        if self.mask_output_dtype is not None:
            sample.mask = torch.from_numpy(np.array(sample.mask)).to(dtype=self.mask_output_dtype)

        # Add dummy channels dimension if needed
        if self.add_mask_dummy_dim and len(sample.mask.shape) == 2:
            sample.mask = sample.mask.unsqueeze(0)

        return sample

    def get_equivalent_preprocessing(self):
        return [{Processings.ImagePermute: {"permutation": (2, 0, 1)}}]


@register_transform(Transforms.SegStandardize)
class SegStandardize(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    """
    Standardize image pixel values with img/max_val

    :param max_value: Current maximum value of the image pixels. (usually 255)
    """

    def __init__(self, max_value=255):
        self.max_value = max_value

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        if not isinstance(sample.image, np.ndarray):
            sample.image = np.array(sample.image)
        sample.image = (sample.image / self.max_value).astype(np.float32)
        return sample

    def get_equivalent_preprocessing(self):
        return [{Processings.StandardizeImage: {"max_value": self.max_value}}]


@register_transform(Transforms.SegNormalize)
class SegNormalize(AbstractSegmentationTransform, LegacySegmentationTransformMixin):
    """
    Normalization to be applied on the segmentation sample's image.

    :param mean (sequence): Sequence of means for each channel.
    :param std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.array(mean).reshape(1, 1, -1).astype(np.float32)
        self.std = np.array(std).reshape(1, 1, -1).astype(np.float32)

    def apply_to_sample(self, sample: SegmentationSample) -> SegmentationSample:
        if not isinstance(sample.image, np.ndarray):
            sample.image = np.array(sample.image)
        sample.image = (sample.image - self.mean) / self.std
        return sample

    def get_equivalent_preprocessing(self):
        return [{Processings.NormalizeImage: {"mean": self.mean, "std": self.std}}]


@register_transform(Transforms.DetectionImagePermute)
class DetectionImagePermute(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Permute image dims. Useful for converting image from HWC to CHW format.
    """

    def __init__(self, dims: Tuple[int, int, int] = (2, 0, 1)):
        """

        :param dims: Specify new order of dims. Default value (2, 0, 1) suitable for converting from HWC to CHW format.
        """
        super().__init__()
        self.dims = tuple(dims)

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        sample.image = np.ascontiguousarray(sample.image.transpose(*self.dims))
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.ImagePermute: {"permutation": self.dims}}]


@register_transform(Transforms.DetectionPadToSize)
class DetectionPadToSize(DetectionPadIfNeeded):
    """
    Preprocessing transform to pad image and bboxes to `input_dim` shape (rows, cols).
    Transform does center padding, so that input image with bboxes located in the center of the produced image.

    Note: This transformation assume that dimensions of input image is equal or less than `output_size`.
    This class exists for backward compatibility with previous versions of the library.
    Use `DetectionPadIfNeeded` instead.
    """

    def __init__(self, output_size: Union[int, Tuple[int, int], None], pad_value: Union[int, Tuple[int, ...]]):
        """
        Constructor for DetectionPadToSize transform.

        :param output_size: Output image size (rows, cols)
        :param pad_value: Padding value for image
        """
        min_height, min_width = ensure_is_tuple_of_two(output_size)

        super().__init__(min_height=min_height, min_width=min_width, pad_value=pad_value, padding_mode="center")
        self.output_size = ensure_is_tuple_of_two(output_size)
        self.pad_value = pad_value


@register_transform(Transforms.DetectionPaddedRescale)
class DetectionPaddedRescale(AbstractDetectionTransform, LegacyDetectionTransformMixin):
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

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        sample.image, r = _rescale_and_pad_to_size(sample.image, self.input_dim, self.swap, self.pad_value)
        sample.bboxes_xyxy = _rescale_xyxy_bboxes(sample.bboxes_xyxy, r)
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [
            {Processings.DetectionLongestMaxSizeRescale: {"output_shape": self.input_dim}},
            {Processings.DetectionBottomRightPadding: {"output_shape": self.input_dim, "pad_value": self.pad_value}},
            {Processings.ImagePermute: {"permutation": self.swap}},
        ]


@register_transform(Transforms.DetectionHorizontalFlip)
class DetectionHorizontalFlip(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Horizontal Flip for Detection

    :param prob:        Probability of applying horizontal flip
    """

    def __init__(self, prob: float):
        super(DetectionHorizontalFlip, self).__init__()
        self.prob = float(prob)

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        """
        Apply horizontal flip to sample
        :param sample: Input detection sample
        :return:       Transformed detection sample
        """
        if random.random() < self.prob:
            sample = DetectionSample(
                image=_flip_horizontal_image(sample.image),
                bboxes_xyxy=_flip_horizontal_boxes_xyxy(sample.bboxes_xyxy, sample.image.shape[1]),
                labels=sample.labels,
                is_crowd=sample.is_crowd,
                additional_samples=None,
            )
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.DetectionVerticalFlip)
class DetectionVerticalFlip(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Vertical Flip for Detection

    :param prob:        Probability of applying vertical flip
    """

    def __init__(self, prob: float):
        super(DetectionVerticalFlip, self).__init__()
        self.prob = float(prob)

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if random.random() < self.prob:
            sample = DetectionSample(
                image=_flip_vertical_image(sample.image),
                bboxes_xyxy=_flip_vertical_boxes_xyxy(sample.bboxes_xyxy, sample.image.shape[0]),
                labels=sample.labels,
                is_crowd=sample.is_crowd,
                additional_samples=None,
            )
        return sample

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.DetectionRescale)
class DetectionRescale(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Resize image and bounding boxes to given image dimensions without preserving aspect ratio

    :param output_shape: (rows, cols)
    """

    def __init__(self, output_shape: Union[int, Tuple[int, int]]):
        super().__init__()
        self.output_shape = ensure_is_tuple_of_two(output_shape)

    @classmethod
    def apply_to_image(self, image: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
        return _rescale_image(image=image, target_shape=(output_height, output_width))

    @classmethod
    def apply_to_bboxes(self, bboxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
        return _rescale_bboxes(bboxes, scale_factors=(sy, sx))

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        image_height, image_width = sample.image.shape[:2]
        output_height, output_width = self.output_shape
        sx = output_width / image_width
        sy = output_height / image_height

        return DetectionSample(
            image=self.apply_to_image(sample.image, output_width=output_width, output_height=output_height),
            bboxes_xyxy=self.apply_to_bboxes(sample.bboxes_xyxy, sx=sx, sy=sy),
            labels=sample.labels,
            is_crowd=sample.is_crowd,
            additional_samples=None,
        )

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.DetectionRescale: {"output_shape": self.output_shape}}]


@register_transform(Transforms.DetectionRandomRotate90)
class DetectionRandomRotate90(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if random.random() < self.prob:
            k = random.randrange(0, 4)
            image_shape = sample.image.shape[:2]
            sample = DetectionSample(
                image=self.apply_to_image(sample.image, k),
                bboxes_xyxy=self.apply_to_bboxes(sample.bboxes_xyxy, k, image_shape),
                labels=sample.labels,
                is_crowd=sample.is_crowd,
                additional_samples=None,
            )
        return sample

    def apply_to_image(self, image: np.ndarray, factor: int) -> np.ndarray:
        """
        Apply a `factor` number of 90-degree rotation to image.

        :param image:  Input image (HWC).
        :param factor: Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        :return:       Rotated image (HWC).
        """
        return np.ascontiguousarray(np.rot90(image, factor))

    def apply_to_bboxes(self, bboxes: np.ndarray, factor: int, image_shape: Tuple[int, int]):
        """
        Apply a `factor` number of 90-degree rotation to bounding boxes.

        :param bboxes:       Input bounding boxes in XYXY format.
        :param factor:       Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        :param image_shape:  Original image shape
        :return:             Rotated bounding boxes in XYXY format.
        """
        rows, cols = image_shape
        bboxes_rotated = self.xyxy_bbox_rot90(bboxes, factor, rows, cols)
        return bboxes_rotated

    @classmethod
    def xyxy_bbox_rot90(cls, bboxes: np.ndarray, factor: int, rows: int, cols: int):
        """
        Rotates a bounding box by 90 degrees CCW (see np.rot90)

        :param bboxes:  Tensor made of bounding box tuples (x_min, y_min, x_max, y_max).
        :param factor:  Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
        :param rows:    Image rows of the original image.
        :param cols:    Image cols of the original image.

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

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.DetectionRGB2BGR)
class DetectionRGB2BGR(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Detection change Red & Blue channel of the image

    :param prob: Probability to apply the transform.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = float(prob)

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if len(sample.image.shape) != 3 or sample.image.shape[2] < 3:
            raise ValueError("DetectionRGB2BGR transform expects image to have 3 channels, got input image shape: " + str(sample.image.shape))
        if random.random() < self.prob:
            sample = DetectionSample(
                image=np.ascontiguousarray(sample.image[..., ::-1]),
                bboxes_xyxy=sample.bboxes_xyxy,
                labels=sample.labels,
                is_crowd=sample.is_crowd,
                additional_samples=None,
            )
        return sample

    def get_equivalent_preprocessing(self) -> List:
        if self.prob < 1:
            raise RuntimeError("Cannot set preprocessing pipeline with randomness. Set prob to 1.")
        return [{Processings.ReverseImageChannels: {}}]


@register_transform(Transforms.DetectionHSV)
class DetectionHSV(AbstractDetectionTransform, LegacyDetectionTransformMixin):
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

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        if sample.image.shape[2] < 3:
            raise ValueError("HSV transform expects at least 3 channels, got: " + str(sample.image.shape[2]))
        if sample.image.shape[2] > 3 and not self._additional_channels_warned:
            logger.warning(
                "HSV transform received image with "
                + str(sample.image.shape[2])
                + " channels. HSV transform will only be applied on channels: "
                + str(self.bgr_channels)
                + "."
            )
            self._additional_channels_warned = True

        if random.random() < self.prob:
            sample = DetectionSample(
                image=self.apply_to_image(sample.image),
                bboxes_xyxy=sample.bboxes_xyxy,
                labels=sample.labels,
                is_crowd=sample.is_crowd,
                additional_samples=None,
            )
        return sample

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        return augment_hsv(image.copy(), self.hgain, self.sgain, self.vgain, self.bgr_channels)

    def get_equivalent_preprocessing(self) -> List[Dict]:
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform(Transforms.DetectionNormalize)
class DetectionNormalize(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Normalize image by subtracting mean and dividing by std.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = np.array(list(mean)).reshape((1, 1, -1)).astype(np.float32)
        self.std = np.array(list(std)).reshape((1, 1, -1)).astype(np.float32)

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        return DetectionSample(
            image=self.apply_to_image(sample.image),
            bboxes_xyxy=sample.bboxes_xyxy,
            labels=sample.labels,
            is_crowd=sample.is_crowd,
            additional_samples=None,
        )

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        return (image - self.mean) / self.std

    def get_equivalent_preprocessing(self) -> List[Dict]:
        return [{Processings.NormalizeImage: {"mean": self.mean, "std": self.std}}]


@register_transform(Transforms.DetectionTargetsFormatTransform)
class DetectionTargetsFormatTransform(AbstractDetectionTransform, LegacyDetectionTransformMixin):
    """
    Detection targets format transform

    Convert targets in input_format to output_format, filter small bboxes and pad targets.

    :param input_dim:          Shape of the images to transform.
    :param input_format:       Format of the input targets. For instance [xmin, ymin, xmax, ymax, cls_id] refers to XYXY_LABEL.
    :param output_format:      Format of the output targets. For instance [xmin, ymin, xmax, ymax, cls_id] refers to XYXY_LABEL
    :param min_bbox_edge_size: bboxes with edge size lower than this values will be removed.
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

    def __call__(self, sample: Union[dict, DetectionSample]) -> dict:
        if isinstance(sample, DetectionSample):
            pass
        else:
            # if self.input_dim not set yet, it will be set with first batch
            if self.input_dim is None:
                self._setup_input_dim_related_params(input_dim=sample["image"].shape[1:])

            sample["target"] = self.apply_on_targets(sample["target"])
            if "crowd_target" in sample.keys():
                sample["crowd_target"] = self.apply_on_targets(sample["crowd_target"])
        return sample

    def apply_to_sample(self, sample: DetectionSample):
        # DIRTY HACK: No-op if a detection sample is passed
        # DIRTY HACK: As a workaround we will do this transform in dataset class for now
        return sample

    def apply_on_targets(self, targets: np.ndarray) -> np.ndarray:
        """Convert targets in input_format to output_format, filter small bboxes and pad targets"""
        targets = self.filter_small_bboxes(targets)
        targets = self.targets_format_converter(targets)
        return np.ascontiguousarray(targets, dtype=np.float32)

    def filter_small_bboxes(self, targets: np.ndarray) -> np.ndarray:
        """Filter bboxes smaller than specified threshold."""

        def _is_big_enough(bboxes: np.ndarray) -> np.ndarray:
            bboxes_xywh = xyxy_to_xywh(bboxes, image_shape=None)
            return np.minimum(bboxes_xywh[:, 2], bboxes_xywh[:, 3]) > self.min_bbox_edge_size

        targets = filter_on_bboxes(fn=_is_big_enough, tensor=targets, tensor_format=self.input_format)
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
    if num_gts == 0:
        return targets
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
    crowd_targets: np.ndarray = None,
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
    :param crowd_targets: Optional array of crowd annotations. If provided, it will be transformed in the same way as targets.
    :return:            Image and Target with applied random affine
    """

    M = get_affine_matrix(img.shape[:2], target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(border_value, border_value, border_value))

    # Transform label coordinates
    if len(targets) > 0:
        targets_seg = np.zeros((targets.shape[0], 0)) if targets_seg is None else targets_seg
        targets_orig = targets.copy()
        targets = apply_affine_to_bboxes(targets, targets_seg, target_size, M)
        if filter_box_candidates:
            box_candidates_ids = _filter_box_candidates(targets_orig[:, :4], targets[:, :4], wh_thr=wh_thr, ar_thr=ar_thr, area_thr=area_thr)
            targets = targets[box_candidates_ids]

    if crowd_targets is not None:
        if len(crowd_targets) > 0:
            crowd_targets_seg = np.zeros((crowd_targets.shape[0], 0))
            crowd_targets_orig = crowd_targets.copy()
            crowd_targets = apply_affine_to_bboxes(crowd_targets, crowd_targets_seg, target_size, M)
            if filter_box_candidates:
                box_candidates_ids = _filter_box_candidates(crowd_targets_orig[:, :4], crowd_targets[:, :4], wh_thr=wh_thr, ar_thr=ar_thr, area_thr=area_thr)
                crowd_targets = crowd_targets[box_candidates_ids]
        return img, targets, crowd_targets

    return img, targets


def _filter_box_candidates(original_bboxes: np.ndarray, transformed_bboxes: np.ndarray, wh_thr=2, ar_thr=20, area_thr=0.1) -> np.ndarray:
    """
    Filter out transformed bboxes by edge size, area ratio, and aspect ratio.

    :param original_bboxes:    Input bboxes in XYXY format of [N,4] shape
    :param transformed_bboxes: Transformed bboxes in XYXY format of [N,4] shape
    :param wh_thr:             Size threshold (Boxes with width or height smaller than this values will be filtered out)
    :param ar_thr:             Aspect ratio threshold (Boxes with aspect ratio larger than this values will be filtered out)
    :param area_thr:           Area threshold (Boxes with area ratio smaller than this value will be filtered out)
    :return:                   A boolean mask of [N] shape indicating which bboxes to keep.
    """
    original_bboxes = original_bboxes.T
    transformed_bboxes = transformed_bboxes.T
    w1, h1 = original_bboxes[2] - original_bboxes[0], original_bboxes[3] - original_bboxes[1]
    w2, h2 = transformed_bboxes[2] - transformed_bboxes[0], transformed_bboxes[3] - transformed_bboxes[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def _flip_horizontal_image(image: np.ndarray) -> np.ndarray:
    """
    Horizontally flips image
    :param image: image to be flipped.
    :return: flipped_image
    """
    return image[:, ::-1]


def _flip_horizontal_boxes_xywh(boxes: np.ndarray, img_width: int) -> np.ndarray:
    """
    Horizontally flips bboxes in XYWH format
    The function modifies the input array in place, and returns it.

    :param boxes:     Input bboxes in XYWH format of [..., 4] shape.
    :param img_width: Image width
    :return:          Output bboxes in XYWH format of [..., 4] shape.
    """
    boxes[..., 0] = img_width - (boxes[..., 0] + boxes[..., 2])
    return boxes


def _flip_horizontal_boxes_xyxy(boxes: np.ndarray, img_width: int) -> np.ndarray:
    """
    Horizontally flips bboxes in XYXY format.
    The function modifies the input array in place, and returns it.

    :param boxes: Input boxes in XYXY format of [..., 4] shape.
    :param img_width: Image width
    :return:          Output bboxes in XYXY format of [..., 4] shape.
    """
    boxes[..., [0, 2]] = img_width - boxes[..., [2, 0]]
    return boxes


def _flip_vertical_image(image: np.ndarray) -> np.ndarray:
    """
    Vertically flips image
    :param image: image to be flipped.
    :return: flipped_image
    """
    return image[::-1, :]


def _flip_vertical_boxes_xyxy(boxes: np.ndarray, img_height: int) -> np.ndarray:
    """
    Vertically flips bboxes. The function modifies the input array in place, and returns it.

    :param boxes: Input bboxes to be flipped in XYXY format of [..., 4] shape
    :return:      Vertically flipped boxes in XYXY format of [..., 4] shape
    """
    boxes[..., [1, 3]] = img_height - boxes[..., [3, 1]]
    return boxes


def _flip_vertical_boxes_xywh(boxes: np.ndarray, img_height: int) -> np.ndarray:
    """
    Vertically flips bboxes. The function modifies the input array in place, and returns it.

    :param boxes: Input bboxes to be flipped in XYWH format of [..., 4] shape
    :return:      Vertically flipped boxes in XYWH format of [..., 4] shape
    """
    boxes[..., 1] = img_height - (boxes[..., 1] + boxes[..., 3])
    return boxes


def augment_hsv(img: np.ndarray, hgain: float, sgain: float, vgain: float, bgr_channels=(0, 1, 2)):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img[..., bgr_channels], cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    img[..., bgr_channels] = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)  # no return needed
    return img


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
