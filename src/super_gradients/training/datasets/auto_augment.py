""" RandAugment
RandAugment is a variant of AutoAugment which randomly selects transformations
 from AutoAugment to be applied on an image.

RandomAugmentation Implementation adapted from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py

Papers:
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719

"""
import random
import re
from typing import List
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

from super_gradients.common.object_names import Transforms
from super_gradients.common.registry.registry import register_transform

_FILL = (128, 128, 128)

# to unify the calls of the different augmentations in terms of params, all augmentations are set to work with a single
# magnitude params, normalized according to _MAX_MAGNITUDE
_MAX_MAGNITUDE = 10.0

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

# Define the interpolation types
_RANDOM_INTERPOLATION = Image.BILINEAR


def _interpolation(kwargs):
    """
    Performs Bi-Linear interpolation
    """
    interpolation = kwargs.pop("resample", Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def shear_x(img, factor, **kwargs):
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    return img.rotate(degrees, **kwargs)


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_MAGNITUDE) * 30.0
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return ((level / _MAX_MAGNITUDE) * 1.8 + 0.1,)


def _enhance_increasing_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    level = (level / _MAX_MAGNITUDE) * 0.9
    level = 1.0 + _randomly_negate(level)
    return (level,)


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_MAGNITUDE) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams["translate_const"]
    level = (level / _MAX_MAGNITUDE) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.45, 0.45]
    translate_pct = hparams.get("translate_pct", 0.45)
    level = (level / _MAX_MAGNITUDE) * translate_pct
    level = _randomly_negate(level)
    return (level,)


def _posterize_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_MAGNITUDE) * 4),)


def _posterize_increasing_level_to_arg(level, hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image',
    # intensity/severity of augmentation increases with level
    return (4 - _posterize_level_to_arg(level, hparams)[0],)


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_MAGNITUDE) * 4) + 4,)


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation decreases with level
    return (int((level / _MAX_MAGNITUDE) * 256),)


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 256]
    # intensity/severity of augmentation increases with level
    return (256 - _solarize_level_to_arg(level, _hparams)[0],)


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return (int((level / _MAX_MAGNITUDE) * 110),)


LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "SolarizeAdd": _solarize_add_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_abs_level_to_arg,
    "TranslateY": _translate_abs_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "SolarizeAdd": solarize_add,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x_abs,
    "TranslateY": translate_y_abs,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}


class AugmentOp:
    """
    single auto augment operations
    """

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams["img_mean"] if "img_mean" in hparams else _FILL,
            resample=hparams["interpolation"] if "interpolation" in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, introduce some randomness
        self.magnitude_std = self.hparams.get("magnitude_std", 0)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std:
            if self.magnitude_std == float("inf"):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_MAGNITUDE, max(0, magnitude))  # clip to valid range
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


_RAND_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "SolarizeAdd",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
    # 'Cutout'  # NOTE I've implement this as random erasing separately
]


_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "SolarizeAdd",
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
    # 'Cutout'  # NOTE I've implement this as random erasing separately
]

# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateXRel": 0.1,
    "TranslateYRel": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "SolarizeAdd": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0,
    "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    """
    Random auto augment class, will select auto augment transforms according to probability weights for each op
    """

    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img


@register_transform(Transforms.RandAugmentTransform)
def rand_augment_transform(config_str, crop_size: int, img_mean: List[float]):
    """
    Create a RandAugment transform

    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied
        'inc' - integer (bool), use augmentations that increase in severity with magnitude (default: 0)
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2

    :param crop_size: The size of crop image
    :param img_mean:  Average per channel

    :return: A PyTorch compatible Transform
    """
    hparams = dict(translate_const=int(crop_size * 0.45), img_mean=tuple([min(255, round(255 * channel_mean)) for channel_mean in img_mean]))

    magnitude = _MAX_MAGNITUDE  # default to _MAX_MAGNITUDE for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    config = config_str.split("-")
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # noise param injected via hparams for now
            hparams.setdefault("magnitude_std", float(val))
        elif key == "inc":
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == "m":
            magnitude = int(val)
        elif key == "n":
            num_layers = int(val)
        elif key == "w":
            weight_idx = int(val)
        else:
            assert False, "Unknown RandAugment config section"
    ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)
