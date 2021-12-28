import random
from PIL import Image, ImageOps, ImageFilter
import collections
from typing import Optional, Union, Tuple, List
import math
import torchvision.transforms as transforms

# FIXME: REFACTOR AUGMENTATIONS, CONSIDER USING A MORE EFFICIENT LIBRARIES SUCH AS, IMGAUG, DALI ETC.

image_resample = Image.BILINEAR
mask_resample = Image.NEAREST


class RandomFlip:
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


class Rescale:
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


class RandomRotate:
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


class CropImageAndMask:
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


class RandomGaussianBlur:
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


class PadShortToCropSize:
    """
    Pads image to 'crop_size'.
    Should be called only after "Rescale" or "RandomRescale" in augmentations pipeline.
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
        self.fill_image = fill_image

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


def coco_sub_classes_inclusion_tuples_list():
    return [(0, 'background'), (5, 'airplane'), (2, 'bicycle'), (16, 'bird'),
            (9, 'boat'),
            (44, 'bottle'), (6, 'bus'), (3, 'car'), (17, 'cat'), (62, 'chair'),
            (21, 'cow'),
            (67, 'dining table'), (18, 'dog'), (19, 'horse'), (4, 'motorcycle'),
            (1, 'person'),
            (64, 'potted plant'), (20, 'sheep'), (63, 'couch'), (7, 'train'),
            (72, 'tv')]
