from typing import Tuple, Union

import numpy as np
from torch import Tensor

from super_gradients.training.utils.tensor_formats.bbox_formats import convert_bboxes
from super_gradients.training.utils.tensor_formats.formats import ConcatenatedTensorFormat, apply_on_bboxes, get_permutation_indexes

__all__ = ["ConcatenatedTensorFormatConverter"]


class ConcatenatedTensorFormatConverter:
    def __init__(
        self,
        input_format: ConcatenatedTensorFormat,
        output_format: ConcatenatedTensorFormat,
        image_shape: Union[Tuple[int, int], None],
    ):
        """
        Adapter class that converts concatenated tensors from input format to output format.

        :param input_format: Format definition of the inputs
        :param output_format: Format definition of the outputs
        :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
                            If you're not using normalized coordinates you can set this to None
        """
        self.permutation_indexes = get_permutation_indexes(input_format, output_format)

        self.input_format = input_format
        self.output_format = output_format
        self.image_shape = image_shape
        self.input_length = input_format.num_channels

    def __call__(self, tensor: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        if tensor.shape[-1] != self.input_length:
            raise RuntimeError(
                f"Number of channels in last dimension of input tensor ({tensor.shape[-1]}) must be "
                f"equal to {self.input_length} as defined by input format."
            )
        tensor = tensor[:, self.permutation_indexes]
        tensor = apply_on_bboxes(fn=self._convert_bbox, tensor=tensor, tensor_format=self.output_format)
        return tensor

    def _convert_bbox(self, bboxes: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        return convert_bboxes(
            bboxes=bboxes,
            source_format=self.input_format.bboxes_format.format,
            target_format=self.output_format.bboxes_format.format,
            inplace=False,
            image_shape=self.image_shape,
        )
