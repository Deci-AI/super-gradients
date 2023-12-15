from typing import Tuple, Optional, Union
from dataclasses import dataclass
import cv2
import numbers
import typing
import numpy as np


@dataclass
class PaddingCoordinates:
    top: int
    bottom: int
    left: int
    right: int


def _rescale_image(image: np.ndarray, target_shape: Tuple[int, int], interpolation_method: Optional = cv2.INTER_LINEAR) -> np.ndarray:
    """Rescale image to target_shape, without preserving aspect ratio.

    :param image:           Image to rescale. (H, W, C) or (H, W).
    :param target_shape:    Target shape to rescale to (H, W).
    :return:                Rescaled image.
    """
    height, width = target_shape[:2]
    return cv2.resize(image, dsize=(width, height), interpolation=interpolation_method)


def _rescale_image_with_pil(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Rescale image to target_shape, without preserving aspect ratio using PIL.
    OpenCV and PIL has slightly different implementations of interpolation methods.
    OpenCV has faster resizing, however PIL is more accurate (not introducing aliasing artifacts).
    We use this method in some preprocessing transforms where we want to keep the compatibility with
    torchvision transforms.

    :param image:           Image to rescale. (H, W, C) or (H, W).
    :param target_shape:    Target shape to rescale to (H, W).
    :return:                Rescaled image.
    """
    height, width = target_shape[:2]
    from PIL import Image

    return np.array(Image.fromarray(image).resize((width, height), Image.BILINEAR))


def _rescale_bboxes(targets: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
    """Rescale bboxes to given scale factors, without preserving aspect ratio.
    This function supports both xyxy and xywh bboxes.

    :param targets:         Targets to rescale (N, 4+), where target[:, :4] is the bounding box coordinates.
    :param scale_factors:   Tuple of (scale_factor_h, scale_factor_w) scale factors to rescale to.
    :return:                Rescaled targets.
    """

    targets = targets.astype(np.float32, copy=True)

    sy, sx = scale_factors
    targets[:, :4] *= np.array([[sx, sy, sx, sy]], dtype=targets.dtype)
    return targets


def _rescale_keypoints(targets: np.ndarray, scale_factors: Tuple[float, float]) -> np.ndarray:
    """Rescale keypoints to given scale factors, without preserving aspect ratio.

    :param targets:         Array of keypoints to rescale. Can have arbitrary shape [N,2], [N,K,2], etc.
                            Last dimension encodes XY coordinates: target[..., 0] is the X coordinates and
                            targets[..., 1] is the Y coordinate.
    :param scale_factors:   Tuple of (scale_factor_h, scale_factor_w) scale factors to rescale to.
    :return:                Rescaled targets.
    """

    targets = targets.astype(np.float32, copy=True)

    sy, sx = scale_factors
    targets[..., 0] *= sx
    targets[..., 1] *= sy
    return targets


def _get_center_padding_coordinates(input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> PaddingCoordinates:
    """Get parameters for padding an image to given output shape, in center mode.

    :param input_shape:  Shape of the input image.
    :param output_shape: Shape to resize to.
    :return:             Padding parameters.
    """
    pad_height, pad_width = output_shape[0] - input_shape[0], output_shape[1] - input_shape[1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return PaddingCoordinates(top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right)


def _get_bottom_right_padding_coordinates(input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> PaddingCoordinates:
    """Get parameters for padding an image to given output shape, in bottom right mode
    (i.e. image will be at top-left while bottom-right corner will be padded).

    :param input_shape:  Shape of the input image.
    :param output_shape: Shape to resize to.
    :return:             Padding parameters.
    """
    pad_height, pad_width = output_shape[0] - input_shape[0], output_shape[1] - input_shape[1]
    return PaddingCoordinates(top=0, bottom=pad_height, left=0, right=pad_width)


def _pad_image(image: np.ndarray, padding_coordinates: PaddingCoordinates, pad_value: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """Pad an image.

    :param image:       Image to shift. (H, W, C) or (H, W).
    :param pad_h:       Tuple of (padding_top, padding_bottom).
    :param pad_w:       Tuple of (padding_left, padding_right).
    :param pad_value:   Padding value. Can be a single scalar (Same value for all channels) or a tuple of values.
                        In the latter case, the tuple length must be equal to the number of channels.
    :return:            Image shifted according to padding coordinates.
    """
    pad_h = (padding_coordinates.top, padding_coordinates.bottom)
    pad_w = (padding_coordinates.left, padding_coordinates.right)

    if len(image.shape) == 3:
        _, _, num_channels = image.shape

        if isinstance(pad_value, numbers.Number):
            pad_value = tuple([pad_value] * num_channels)
        else:
            if isinstance(pad_value, typing.Sized) and len(pad_value) != num_channels:
                raise ValueError(f"A pad_value tuple ({pad_value} length should be {num_channels} for an image with {num_channels} channels")

            pad_value = tuple(pad_value)

        constant_values = ((pad_value, pad_value), (pad_value, pad_value), (0, 0))
        # Fixes issue with numpy deprecation warning since constant_values is ragged array (Have to explicitly specify object dtype)
        constant_values = np.array(constant_values, dtype=np.object_)

        padding_values = (pad_h, pad_w, (0, 0))
    else:
        if isinstance(pad_value, numbers.Number):
            pass
        elif isinstance(pad_value, typing.Sized):
            if len(pad_value) != 1:
                raise ValueError(f"A pad_value tuple ({pad_value} length should be 1 for a grayscale image")
            else:
                (pad_value,) = pad_value  # Unpack to a single scalar
        else:
            raise ValueError(f"Unsupported pad_value type {type(pad_value)}")

        constant_values = pad_value
        padding_values = (pad_h, pad_w)

    return np.pad(image, pad_width=padding_values, mode="constant", constant_values=constant_values)


def _shift_bboxes_xyxy(targets: np.array, shift_w: float, shift_h: float) -> np.ndarray:
    """Shift bboxes with respect to padding values.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    :param shift_w:  shift width.
    :param shift_h:  shift height.
    :return:         Bboxes transformed of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    """
    boxes, labels = targets[:, :4].copy(), targets[:, 4:]
    boxes[:, [0, 2]] += shift_w
    boxes[:, [1, 3]] += shift_h
    return np.concatenate((boxes, labels), 1)


def _shift_keypoints(targets: np.array, shift_w: float, shift_h: float) -> np.ndarray:
    """Shift keypoints with respect to padding values.

    :param targets:  Keypoints to transform of shape (N, 2+), or (N, K, 2+), in format [x1, y1, ...]
    :param shift_w:  shift width.
    :param shift_h:  shift height.
    :return:         Transformed keypoints of the same shape as input.
    """
    targets = targets.copy()
    targets[..., 0] += shift_w
    targets[..., 1] += shift_h
    return targets


def _rescale_xyxy_bboxes(targets: np.ndarray, r: float) -> np.ndarray:
    """Scale targets to given scale factors.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    :param r:        DetectionRescale coefficient that was applied to the image
    :return:         Rescaled Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    """
    return _rescale_bboxes(targets, (r, r))


def _rescale_xywh_bboxes(targets: np.ndarray, r: float) -> np.ndarray:
    """Scale targets to given scale factors.

    :param targets:  Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    :param r:        DetectionRescale coefficient that was applied to the image
    :return:         Rescaled Bboxes to transform of shape (N, 4+), in format [x1, y1, x2, y2, ...]
    """
    return _rescale_bboxes(targets, (r, r))


def _rescale_and_pad_to_size(image: np.ndarray, output_shape: Tuple[int, int], swap: Tuple[int] = (2, 0, 1), pad_val: int = 114) -> Tuple[np.ndarray, float]:
    """
    Rescales image according to minimum ratio input height/width and output height/width rescaled_padded_image,
    pads the image to the target shape and finally swap axis.
    Note: Pads the image to corner, padding is not centered.

    :param image:           Image to be rescaled. (H, W, C) or (H, W).
    :param output_shape:    Target Shape.
    :param swap:            Axis's to be rearranged.
    :param pad_val:         Value to use for padding.
    :return:
        - Rescaled image while preserving aspect ratio, padded to fit output_shape and with axis swapped. By default, (C, H, W).
        - Minimum ratio between the input height/width and output height/width.
    """
    r = min(output_shape[0] / image.shape[0], output_shape[1] / image.shape[1])
    rescale_shape = (int(image.shape[0] * r), int(image.shape[1] * r))

    resized_image = _rescale_image(image=image, target_shape=rescale_shape)

    padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=rescale_shape, output_shape=output_shape)
    padded_image = _pad_image(image=resized_image, padding_coordinates=padding_coordinates, pad_value=pad_val)

    padded_image = padded_image.transpose(swap)
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, r


def _compute_scale_factor(scale_factor: float, short_size: int, long_size: int, image_width: int, image_height: int):
    """
    Calculates rescale factor for SegResacle transform (and the equivalent processing).
    The rescaling can be done according to scale_factor, short_size or long_size.
    If more than one argument is given, the rescaling factor is determined by this order: scale_factor, then short_size,
    then long_size.

    :param scale_factor: Rescaling is done  in "SegRescale" by multiplying input size by scale_factor:
            out_size = (scale_factor * w, scale_factor * h)
    :param short_size:  Rescaling is done by determining the scale factor by the ratio short_size / min(h, w).
    :param long_size:   Rescaling is done by determining the scale factor by the ratio long_size / max(h, w).
    :param image_width: W
    :param image_height: H
    :return:
        - Rescaling factor to be used by the transform / processing.
    """
    if scale_factor is not None:
        scale = scale_factor
    elif short_size is not None:
        img_short_size = min(image_width, image_height)
        scale = short_size / img_short_size
    else:
        img_long_size = max(image_width, image_height)
        scale = long_size / img_long_size
    return scale
