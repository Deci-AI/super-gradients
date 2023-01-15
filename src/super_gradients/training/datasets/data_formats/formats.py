import collections
from typing import Tuple, Union, List, Mapping, Callable

import numpy as np
from torch import Tensor

from super_gradients.training.datasets.data_formats.bbox_formats import BoundingBoxFormat


class DetectionOutputFormat:
    pass


class TensorSliceItem:
    length: int
    name: str

    def __init__(self, name: str, length: int):
        self.name = name
        self.length = length

    def __repr__(self):
        return f"name={self.name} length={self.length}"


class BoundingBoxesTensorSliceItem(TensorSliceItem):
    format: BoundingBoxFormat

    def __init__(self, name: str, format: BoundingBoxFormat):
        super().__init__(name, length=format.get_num_parameters())
        self.format = format

    def __repr__(self):
        return f"name={self.name} length={self.length} format={self.format}"


class ConcatenatedTensorFormat(DetectionOutputFormat):
    """
    Define the output format that return a single tensor of shape [N,M] (N - number of detections,
    M - sum of bbox attributes) that is a concatenated from bbox coordinates and other fields.
    A layout defines the order of concatenated tensors. For instance:
    - layout: (bboxes, scores, labels) gives a Tensor that is product of torch.cat([bboxes, scores, labels], dim=1)
    - layout: (labels, bboxes) produce a Tensor from torch.cat([labels, bboxes], dim=1)


    >>> from super_gradients.training.datasets.data_formats.formats import ConcatenatedTensorFormat, BoundingBoxesTensorSliceItem, TensorSliceItem
    >>> from super_gradients.training.datasets.data_formats.bbox_formats import XYXYCoordinateFormat, NormalizedXYWHCoordinateFormat
    >>>
    >>> custom_format = ConcatenatedTensorFormat(
    >>>     layout=(
    >>>         BoundingBoxesTensorSliceItem(name="bboxes", format=XYXYCoordinateFormat()),
    >>>         TensorSliceItem(name="label", length=1),
    >>>         TensorSliceItem(name="distance", length=1),
    >>>         TensorSliceItem(name="attributes", length=4),
    >>>     )
    >>> )

    """

    layout: Mapping[str, TensorSliceItem]
    locations: Mapping[str, Tuple[int, int]]
    indexes: Mapping[str, List[int]]
    num_channels: int

    @property
    def bboxes_format(self) -> BoundingBoxesTensorSliceItem:
        bbox_items = [x for x in self.layout.values() if isinstance(x, BoundingBoxesTensorSliceItem)]
        return bbox_items[0]

    def __init__(self, layout: Union[List[TensorSliceItem], Tuple[TensorSliceItem, ...]]):
        bbox_items = [x for x in layout if isinstance(x, BoundingBoxesTensorSliceItem)]
        if len(bbox_items) != 1:
            raise RuntimeError("Number of bounding box items must be strictly equal to 1")

        _layout = []
        _locations = []
        _indexes = []

        offset = 0
        for item in layout:
            location_indexes = list(range(offset, offset + item.length))
            location_slice = offset, offset + item.length

            _layout.append((item.name, item))
            _locations.append((item.name, location_slice))
            _indexes.append((item.name, location_indexes))
            offset += item.length

        self.layout = collections.OrderedDict(_layout)
        self.locations = collections.OrderedDict(_locations)
        self.indexes = collections.OrderedDict(_indexes)
        self.num_channels = offset

    def __repr__(self):
        return str(self.layout)


def apply_on_bboxes(
    fn: Callable[[Union[np.ndarray, Tensor]], Union[np.ndarray, Tensor]],
    tensor: Union[np.ndarray, Tensor],
    tensor_format: ConcatenatedTensorFormat,
) -> Union[np.ndarray, Tensor]:
    """Apply inplace a function only on the bboxes of a concatenated tensor.

    :param fn:              Function to apply on the bboxes.
    :param tensor:          Concatenated tensor that include - among other - the bboxes.
    :param tensor_format:   Format of the tensor, required to know the indexes of the bboxes.
    :return:                Tensor, after applying INPLACE the fn on the bboxes
    """
    return apply_on_layout(fn=fn, tensor=tensor, tensor_format=tensor_format, layout_name=tensor_format.bboxes_format.name)


def apply_on_layout(
    fn: Callable[[Union[np.ndarray, Tensor]], Union[np.ndarray, Tensor]],
    tensor: Union[np.ndarray, Tensor],
    tensor_format: ConcatenatedTensorFormat,
    layout_name: str,
) -> Union[np.ndarray, Tensor]:
    """Apply inplace a function only on a specific layout of a concatenated tensor.
    :param fn:              Function to apply on the bboxes.
    :param tensor:          Concatenated tensor that include - among other - the layout of interest.
    :param tensor_format:   Format of the tensor, required to know the indexes of the layout.
    :param layout_name:     Name of the layout of interest. It has to be defined in the tensor_format.
    :return:                Tensor, after applying INPLACE the fn on the layout
    """
    location = slice(*iter(tensor_format.locations[layout_name]))
    result = fn(tensor[..., location])
    tensor[..., location] = result
    return tensor


def filter_on_bboxes(
    fn: Callable[[Union[np.ndarray, Tensor]], Union[np.ndarray, Tensor]],
    tensor: Union[np.ndarray, Tensor],
    tensor_format: ConcatenatedTensorFormat,
) -> Union[np.ndarray, Tensor]:
    """Filter the tensor according to a condition on the bboxes.

    :param fn:              Function to filter the bboxes (keep only True elements).
    :param tensor:          Concatenated tensor that include - among other - the bboxes.
    :param tensor_format:   Format of the tensor, required to know the indexes of the bboxes.
    :return:                Tensor, after applying INPLACE the fn on the bboxes
    """
    return filter_on_layout(fn=fn, tensor=tensor, tensor_format=tensor_format, layout_name=tensor_format.bboxes_format.name)


def filter_on_layout(
    fn: Callable[[Union[np.ndarray, Tensor]], Union[np.ndarray, Tensor]],
    tensor: Union[np.ndarray, Tensor],
    tensor_format: ConcatenatedTensorFormat,
    layout_name: str,
) -> Union[np.ndarray, Tensor]:
    """Filter the tensor according to a condition on a specific layout.

    :param fn:              Function to filter the bboxes (keep only True elements).
    :param tensor:          Concatenated tensor that include - among other - the layout of interest.
    :param tensor_format:   Format of the tensor, required to know the indexes of the layout.
    :param layout_name:     Name of the layout of interest. It has to be defined in the tensor_format.
    :return:                Tensor, after filtering the bboxes according to fn.
    """
    location = slice(*tensor_format.locations[layout_name])
    mask = fn(tensor[..., location])
    tensor = tensor[mask]
    return tensor


def get_permutation_indexes(input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat) -> List[int]:
    """Compute the permutations required to change the format layout order.

    :param input_format:    Input format to transform from
    :param output_format:   Output format to transform to
    :return: Permutation indexes to go from input to output format.
    """
    output_indexes = []
    for output_name, output_spec in output_format.layout.items():
        if output_name not in input_format.layout:
            raise KeyError(f"Requested item '{output_name}' was not found among input format spec. Present items are: {tuple(input_format.layout.keys())}")

        input_spec = input_format.layout[output_name]
        if input_spec.length != output_spec.length:
            raise RuntimeError(
                f"Length of the output must match in input and output format. "
                f"Input spec size is {input_spec.length} for key '{output_name}' and output spec size is {output_spec.length}."
            )
        indexes = input_format.indexes[output_name]
        output_indexes.extend(indexes)
    return output_indexes
