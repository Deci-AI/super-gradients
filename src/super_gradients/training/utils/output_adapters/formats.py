import dataclasses
from abc import abstractmethod
from typing import Dict, Tuple, Union, Any

import torch
from torch import Tensor, nn

from super_gradients.training.utils.bbox_formats import BoundingBoxFormat, convert_bboxes


class DetectionOutputFormat:
    @abstractmethod
    def to_dict(self, inputs) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def from_dict(self, values: Dict[str, Any]) -> Any:
        raise NotImplementedError()


@dataclasses.dataclass
class TensorSliceItem:
    location: slice
    name: str
    transform: Union[nn.Module, None] == nn.Identity()

    def get_input(self, input: Tensor, **kwargs):
        return input[..., self.location]

    def get_output(self, values: Tensor, output_format: "TensorSliceItem", **kwargs):
        return self.transform(values)


@dataclasses.dataclass
class BoundingBoxesTensorSliceItem(TensorSliceItem):
    format: BoundingBoxFormat

    def get_output(self, values: Tensor, output_format: "BoundingBoxesTensorSliceItem", **kwargs):
        image_shape = kwargs.get("image_shape", None)
        return convert_bboxes(values, image_shape=image_shape, source_format=output_format.format, target_format=self.format, inplace=False)


# @dataclasses.dataclass
# class ArgmaxTensorSliceItem(TensorSliceItem):
#     def get(self, input: Tensor) -> Tensor:
#         return torch.argmax(super().get(input), dim=-1, keepdim=True)
#
#
# @dataclasses.dataclass
# class MaxScoreTensorSliceItem(TensorSliceItem):
#     def get(self, input: Tensor) -> Tensor:
#         return torch.max(super().get(input), dim=-1, keepdim=True)


@dataclasses.dataclass
class ConcatenatedTensorFormat(DetectionOutputFormat):
    """
    Define the output format that return a single tensor of shape [N,M] (N - number of detections,
    M - sum of bbox attributes) that is a concatenated from bbox coordinates and other fields.
    A layout defines the order of concatenated tensors. For instance:
    - layout: (bboxes, scores, labels) gives a Tensor that is product of torch.cat([bboxes, scores, labels], dim=1)
    - layout: (labels, bboxes) produce a Tensor from torch.cat([labels, bboxes], dim=1)
    """

    layout: Tuple[TensorSliceItem, ...]

    def __init__(self, layout: Tuple[TensorSliceItem, ...]):
        self.layout = layout

    def to_dict(self, inputs: Tensor, **kwargs) -> Dict[TensorSliceItem, Tensor]:
        if not torch.is_tensor(inputs):
            raise RuntimeError(f"Input argument must be a tensor. Got input of type {type(inputs)}")

        named_inputs = {}
        for element in self.layout:
            named_inputs[element] = element.get_input(inputs, **kwargs)

        return named_inputs

    def from_dict(self, values: Dict[TensorSliceItem, Tensor], **kwargs) -> Tensor:
        components = []
        for output_element in self.layout:
            value = values[output_element]
            components.append(output_element.get_output(value, **kwargs))
        return torch.cat(components, dim=-1)
