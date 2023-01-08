import collections
import math
import random
from typing import Optional, Union, Tuple, List, Sequence, Dict

import torch.nn
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms as transforms
import numpy as np
import cv2
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.data_formats_factory import ConcatenatedTensorFormatFactory
from super_gradients.training.utils.detection_utils import get_mosaic_coordinate, adjust_box_anns, xyxy2cxcywh, cxcywh2xyxy
from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormatConverter
from super_gradients.training.datasets.data_formats.formats import filter_on_bboxes, ConcatenatedTensorFormat
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL, LABEL_CXCYWH

image_resample = Image.BILINEAR
mask_resample = Image.NEAREST

logger = get_logger(__name__)


class SegmentationTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + str(self.__dict__).replace("{", "(").replace("}", ")")


class SegResize(SegmentationTransform):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        sample["image"] = image.resize((self.w, self.h), image_resample)
        sample["mask"] = mask.resize((self.w, self.h), mask_resample)
        return sample


class SegRandomFlip(SegmentationTransform):
    """
    Randomly flips the image and mask (synchronously) with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0, f"Probability value must be between 0 and 1, found {prob}"
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


class SegRescale(SegmentationTransform):
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

    def __init__(self, scale_factor: Optional[float] = None, short_size: Optional[int] = None, long_size: Optional[int] = None):
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


class SegRandomRescale:
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
            raise ValueError(f"SegRandomRescale scale values must be positive numbers, found: {self.scales}")
        if self.scales[0] > self.scales[1]:
            self.scales = (self.scales[1], self.scales[0])
        return self.scales


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

    def __call__(self, sample: dict):
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


class SegRandomGaussianBlur(SegmentationTransform):
    """
    Adds random Gaussian Blur to image with probability 'prob'.
    """

    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0, "Probability value must be between 0 and 1"
        self.prob = prob

    def __call__(self, sample: dict):
        image = sample["image"]
        mask = sample["mask"]

        if random.random() < self.prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        sample["image"] = image
        sample["mask"] = mask

        return sample


class SegPadShortToCropSize(SegmentationTransform):
    """
    Pads image to 'crop_size'.
    Should be called only after "SegRescale" or "SegRandomRescale" in augmentations pipeline.
    """

    def __init__(self, crop_size: Union[float, Tuple, List], fill_mask: int = 0, fill_image: Union[int, Tuple, List] = 0):
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
        return self.__class__.__name__ + str(self.__dict__).replace("{", "(").replace("}", ")")


class DetectionMosaic(DetectionTransform):
    """
    DetectionMosaic detection transform

    Attributes:
        input_dim: (tuple) input dimension.
        prob: (float) probability of applying mosaic.
        enable_mosaic: (bool) whether to apply mosaic at all (regardless of prob) (default=True).
        border_value: value for filling borders after applying transforms (default=114).

    """

    def __init__(self, input_dim: tuple, prob: float = 1.0, enable_mosaic: bool = True, border_value=114):
        super(DetectionMosaic, self).__init__(additional_samples_count=3)
        self.prob = prob
        self.input_dim = input_dim
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


class DetectionRandomAffine(DetectionTransform):
    """
    DetectionRandomAffine detection transform

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

     filter_box_candidates: (bool) whether to filter out transformed bboxes by edge size, area ratio, and aspect ratio (default=False).

     wh_thr: (float) edge size threshold when filter_box_candidates = True. Bounding oxes with edges smaller
      then this values will be filtered out. (default=2)

     ar_thr: (float) aspect ratio threshold filter_box_candidates = True. Bounding boxes with aspect ratio larger
      then this values will be filtered out. (default=20)

     area_thr:(float) threshold for area ratio between original image and the transformed one, when when filter_box_candidates = True.
      Bounding boxes with such ratio smaller then this value will be filtered out. (default=0.1)

     border_value: value for filling borders after applying transforms (default=114).


    """

    def __init__(
        self,
        degrees=10,
        translate=0.1,
        scales=0.1,
        shear=10,
        target_size=(640, 640),
        filter_box_candidates: bool = False,
        wh_thr=2,
        ar_thr=20,
        area_thr=0.1,
        border_value=114,
    ):
        super(DetectionRandomAffine, self).__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale = scales
        self.shear = shear
        self.target_size = target_size
        self.enable = True
        self.filter_box_candidates = filter_box_candidates
        self.wh_thr = wh_thr
        self.ar_thr = ar_thr
        self.area_thr = area_thr
        self.border_value = border_value

    def close(self):
        self.enable = False

    def __call__(self, sample: dict):
        if self.enable:
            img, target = random_affine(
                sample["image"],
                sample["target"],
                sample.get("target_seg"),
                target_size=self.target_size,
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


class DetectionMixup(DetectionTransform):
    """
    Mixup detection transform

    Attributes:
        input_dim: (tuple) input dimension.
        mixup_scale: (tuple) scale range for the additional loaded image for mixup.
        prob: (float) probability of applying mixup.
        enable_mixup: (bool) whether to apply mixup at all (regardless of prob) (default=True).
        flip_prob: (float) prbability to apply horizontal flip to the additional sample.
        border_value: value for filling borders after applying transform (default=114).

    """

    def __init__(self, input_dim, mixup_scale, prob=1.0, enable_mixup=True, flip_prob=0.5, border_value=114):
        super(DetectionMixup, self).__init__(additional_samples_count=1, non_empty_targets=True)
        self.input_dim = input_dim
        self.mixup_scale = mixup_scale
        self.prob = prob
        self.enable_mixup = enable_mixup
        self.flip_prob = flip_prob
        self.border_value = border_value

    def close(self):
        self.additional_samples_count = 0
        self.enable_mixup = False

    def __call__(self, sample: dict):
        if self.enable_mixup and random.random() < self.prob:
            origin_img, origin_labels = sample["image"], sample["target"]
            cp_sample = sample["additional_samples"][0]
            img, cp_labels = cp_sample["image"], cp_sample["target"]
            cp_boxes = cp_labels[:, :4]

            img, cp_boxes = _mirror(img, cp_boxes, self.flip_prob)
            # PLUG IN TARGET THE FLIPPED BOXES
            cp_labels[:, :4] = cp_boxes

            jit_factor = random.uniform(*self.mixup_scale)

            if len(img.shape) == 3:
                cp_img = np.ones((self.input_dim[0], self.input_dim[1], img.shape[2]), dtype=np.uint8) * self.border_value
            else:
                cp_img = np.ones(self.input_dim, dtype=np.uint8) * self.border_value

            cp_scale_ratio = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
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


class DetectionPaddedRescale(DetectionTransform):
    """
    Preprocessing transform to be applied last of all transforms for validation.

    Image- Rescales and pads to self.input_dim.
    Targets- pads targets to max_targets, moves the class label to first index, converts boxes format- xyxy -> cxcywh.

    Attributes:
        input_dim: (tuple) final input dimension (default=(640,640))
        swap: image axis's to be rearranged.

    """

    def __init__(self, input_dim, swap=(2, 0, 1), max_targets=50, pad_value=114):
        self.swap = swap
        self.input_dim = input_dim
        self.max_targets = max_targets
        self.pad_value = pad_value

    def __call__(self, sample: Dict[str, np.array]):
        img, targets, crowd_targets = sample["image"], sample["target"], sample.get("crowd_target")
        img, r = rescale_and_pad_to_size(img, self.input_dim, self.swap, self.pad_value)

        sample["image"] = img
        sample["target"] = self._rescale_target(targets, r)
        if crowd_targets is not None:
            sample["crowd_target"] = self._rescale_target(crowd_targets, r)
        return sample

    def _rescale_target(self, targets: np.array, r: float) -> np.array:
        """SegRescale the target according to a coefficient used to rescale the image.
        This is done to have images and targets at the same scale.

        :param targets:  Targets to rescale, shape (batch_size, 6)
        :param r:        SegRescale coefficient that was applied to the image

        :return:         Rescaled targets, shape (batch_size, 6)
        """
        targets = targets.copy() if len(targets) > 0 else np.zeros((self.max_targets, 5), dtype=np.float32)
        boxes, labels = targets[:, :4], targets[:, 4]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r
        boxes = cxcywh2xyxy(boxes)
        return np.concatenate((boxes, labels[:, np.newaxis]), 1)


class DetectionHorizontalFlip(DetectionTransform):
    """
    Horizontal Flip for Detection

    Attributes:
        prob: float: probability of applying horizontal flip
        max_targets: int: max objects in single image, padding target to this size in case of empty image.
    """

    def __init__(self, prob, max_targets: int = 120):
        super(DetectionHorizontalFlip, self).__init__()
        self.prob = prob
        self.max_targets = max_targets

    def __call__(self, sample):
        image, targets = sample["image"], sample["target"]
        boxes = targets[:, :4]
        if len(boxes) == 0:
            targets = np.zeros((self.max_targets, 5), dtype=np.float32)
            boxes = targets[:, :4]
        image, boxes = _mirror(image, boxes, self.prob)
        targets[:, :4] = boxes
        sample["target"] = targets
        sample["image"] = image
        return sample


class DetectionHSV(DetectionTransform):
    """
    Detection HSV transform.

    Attributes:
        prob: (float) probability to apply the transform.
        hgain: (float) hue gain (default=0.5)
        sgain: (float) saturation gain (default=0.5)
        vgain: (float) value gain (default=0.5)
        bgr_channels: (tuple) channel indices of the BGR channels- useful for images with >3 channels,
         or when BGR channels are in different order. (default=(0,1,2)).

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


class DetectionTargetsFormatTransform(DetectionTransform):
    """
    Detection targets format transform

    Convert targets in input_format to output_format, filter small bboxes and pad targets.
    Attributes:
        image_shape:        Shape of the images to transform.
        input_format:       Format of the input targets. For instance [xmin, ymin, xmax, ymax, cls_id] refers to XYXY_LABEL
        output_format:      Format of the output targets. For instance [xmin, ymin, xmax, ymax, cls_id] refers to XYXY_LABEL
        min_bbox_edge_size: bboxes with edge size lower then this values will be removed.
        max_targets:        Max objects in single image, padding target to this size.
    """

    @resolve_param("input_format", ConcatenatedTensorFormatFactory())
    @resolve_param("output_format", ConcatenatedTensorFormatFactory())
    def __init__(
        self,
        image_shape: tuple,
        input_format: ConcatenatedTensorFormat = XYXY_LABEL,
        output_format: ConcatenatedTensorFormat = LABEL_CXCYWH,
        min_bbox_edge_size: float = 1,
        max_targets: int = 120,
    ):
        super(DetectionTargetsFormatTransform, self).__init__()
        self.input_format = input_format
        self.output_format = output_format
        self.max_targets = max_targets
        self.min_bbox_edge_size = min_bbox_edge_size / max(image_shape) if output_format.bboxes_format.format.normalized else min_bbox_edge_size
        self.targets_format_converter = ConcatenatedTensorFormatConverter(input_format=input_format, output_format=output_format, image_shape=image_shape)

    def __call__(self, sample: dict) -> dict:
        sample["target"] = self.apply_on_targets(sample["target"])
        if "crowd_target" in sample.keys():
            sample["crowd_target"] = self.apply_on_targets(sample["crowd_target"])
        return sample

    def apply_on_targets(self, targets: np.ndarray) -> np.ndarray:
        """Convert targets in input_format to output_format, filter small bboxes and pad targets"""
        targets = self.targets_format_converter(targets)
        targets = self.filter_small_bboxes(targets)
        targets = self.pad_targets(targets)
        return targets

    def filter_small_bboxes(self, targets: np.ndarray) -> np.ndarray:
        """Filter bboxes smaller than specified threshold."""

        def _is_big_enough(bboxes: np.ndarray) -> np.ndarray:
            return np.minimum(bboxes[:, 2], bboxes[:, 3]) > self.min_bbox_edge_size

        targets = filter_on_bboxes(fn=_is_big_enough, tensor=targets, tensor_format=self.output_format)
        return targets

    def pad_targets(self, targets: np.ndarray) -> np.ndarray:
        """Pad targets."""
        padded_targets = np.zeros((self.max_targets, targets.shape[-1]))
        padded_targets[range(len(targets))[: self.max_targets]] = targets[: self.max_targets]
        padded_targets = np.ascontiguousarray(padded_targets, dtype=np.float32)
        return padded_targets


def get_aug_params(value: Union[tuple, float], center: float = 0):
    """
    Generates a random value for augmentations as described below

    :param value: Union[tuple, float] defines the range of values for generation. Wen tuple-
     drawn uniformly between (value[0], value[1]), and (center - value, center + value) when float
    :param center: float, defines center to subtract when value is float.
    :return: generated value
    """
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
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
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    """
    Returns a random affine transform matrix.

    :param target_size: (tuple) desired output shape.

    :param degrees:  (Union[tuple, float]) degrees for random rotation, when float the random values are drawn uniformly
     from (-degrees, degrees)

    :param translate:  (Union[tuple, float]) translate size (in pixels) for random translation, when float the random values
     are drawn uniformly from (-translate, translate)

    :param scales: (Union[tuple, float]) values for random rescale, when float the random values are drawn uniformly
     from (0.1-scales, 0.1+scales)

    :param shear: (Union[tuple, float]) degrees for random shear, when float the random values are drawn uniformly
     from (shear, shear)

    :return: affine_transform_matrix, drawn_scale
    """
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


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
        new_bboxes = np.ones((0, 4), dtype=np.float)

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
        new_tight_bboxes = np.ones((0, 4), dtype=np.float)

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
    :param img:         Input image
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
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=border_value)

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


def rescale_and_pad_to_size(img, input_size, swap=(2, 0, 1), pad_val=114):
    """
    Rescales image according to minimum ratio between the target height /image height, target width / image width,
    and pads the image to the target size.

    :param img: Image to be rescaled
    :param input_size: Target size
    :param swap: Axis's to be rearranged.
    :return: rescaled image, ratio
    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], img.shape[-1]), dtype=np.uint8) * pad_val
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * pad_val

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


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
