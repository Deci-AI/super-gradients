from typing import List, Callable, Union

import numpy as np
from torch import Tensor

from super_gradients.training.utils.output_adapters.formats import ConcatenatedTensorFormat


def apply_on_bboxes(fn: Callable, tensor: Union[np.ndarray, Tensor], tensor_format: ConcatenatedTensorFormat) -> Union[np.ndarray, Tensor]:
    """Map inplace!"""
    return apply_on_layout(fn=fn, tensor=tensor, tensor_format=tensor_format, layout_name=tensor_format.bboxes_format.name)


def apply_on_layout(fn: Callable, tensor: Union[np.ndarray, Tensor], tensor_format: ConcatenatedTensorFormat, layout_name: str) -> Union[np.ndarray, Tensor]:
    """Map inplace!"""
    location = slice(*iter(tensor_format.locations[layout_name]))
    result = fn(tensor[..., location])
    tensor[..., location] = result
    return tensor


def filter_on_bboxes(fn: Callable, tensor: Union[np.ndarray, Tensor], tensor_format: ConcatenatedTensorFormat) -> Union[np.ndarray, Tensor]:
    return filter_on_layout(fn=fn, tensor=tensor, tensor_format=tensor_format, layout_name=tensor_format.bboxes_format.name)


def filter_on_layout(fn: Callable, tensor: Union[np.ndarray, Tensor], tensor_format: ConcatenatedTensorFormat, layout_name: str) -> Union[np.ndarray, Tensor]:
    location = slice(*tensor_format.locations[layout_name])
    mask = fn(tensor[..., location])
    tensor = tensor[mask]
    return tensor


def get_permutation_indexes(input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat) -> List[Union[np.ndarray, Tensor]]:
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
