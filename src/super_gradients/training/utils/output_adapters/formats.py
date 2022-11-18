import collections
from abc import abstractmethod
from typing import Dict, Tuple, Union, Any

import torch
from super_gradients.training.utils.bbox_formats import BoundingBoxFormat, convert_bboxes
from torch import Tensor, nn


class DetectionOutputFormat:
    @abstractmethod
    def to_dict(self, inputs) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def from_dict(self, values: Dict[str, Any]) -> Any:
        raise NotImplementedError()


class TensorSliceItem:
    location: slice
    name: str
    transform: Union[nn.Module, None]

    def __init__(self, name: str, location: slice, transorm: Union[nn.Module, None] = None):
        self.name = name
        self.location = location
        self.transform = transorm or nn.Identity()

    def get_input(self, input: Tensor, **kwargs):
        return input[..., self.location]

    def get_output(self, values: Tensor, output_format: "TensorSliceItem", **kwargs):
        return self.transform(values)


class BoundingBoxesTensorSliceItem(TensorSliceItem):
    format: BoundingBoxFormat

    def __init__(self, name: str, location: slice, format: BoundingBoxFormat, transorm: Union[nn.Module, None] = None):
        super().__init__(name, location, transorm)
        self.format = format

    def get_output(self, values: Tensor, output_format: "BoundingBoxesTensorSliceItem", **kwargs):
        image_shape = kwargs.get("image_shape", None)
        return convert_bboxes(values, image_shape=image_shape, source_format=output_format.format, target_format=self.format, inplace=False)


class ConcatenatedTensorFormat(DetectionOutputFormat):
    """
    Define the output format that return a single tensor of shape [N,M] (N - number of detections,
    M - sum of bbox attributes) that is a concatenated from bbox coordinates and other fields.
    A layout defines the order of concatenated tensors. For instance:
    - layout: (bboxes, scores, labels) gives a Tensor that is product of torch.cat([bboxes, scores, labels], dim=1)
    - layout: (labels, bboxes) produce a Tensor from torch.cat([labels, bboxes], dim=1)
    """

    layout: collections.OrderedDict[str, TensorSliceItem]

    def __init__(self, layout: Tuple[TensorSliceItem, ...]):
        self.layout = collections.OrderedDict([(item.name, item) for item in layout])

    def to_dict(self, inputs: Tensor, **kwargs) -> Dict[str, Tensor]:
        if not torch.is_tensor(inputs):
            raise RuntimeError(f"Input argument must be a tensor. Got input of type {type(inputs)}")

        named_inputs: Dict[str, Tensor] = {}
        for name, element in self.layout.items():
            named_inputs[name] = element.get_input(inputs, **kwargs)

        return named_inputs

    def from_dict(self, values: Dict[str, Tensor], **kwargs) -> Tensor:
        components = []
        for name, output_element in self.layout.items():
            value = values[output_element.name]
            components.append(output_element.get_output(value, **kwargs))
        return torch.cat(components, dim=-1)
