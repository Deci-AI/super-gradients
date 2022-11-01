import dataclasses
from typing import Tuple, Union, Callable

import torch
from torch import nn, Tensor

from super_gradients.training.utils.bbox_formats import BoundingBoxFormat, XYWHCoordinateFormat,XYXYCoordinateFormat, YXYXCoordinateFormat, NormalizedXYXYCoordinateFormat, NormalizedXYWHCoordinateFormat


@dataclasses.dataclass
class ConcatenatedTensorDetectionOutputFormat(DetectionOutputFormat):
    """
    Define the output format that return a single tensor of shape [N,M] (N - number of detections,
    M - sum of bbox attributes) that is a concatenated from bbox coordinates and other fields.
    A layout defines the order of concatenated tensors. For instance:
    - layout: (bboxes, scores, labels) gives a Tensor that is product of torch.cat([bboxes, scores, labels], dim=1)
    - layout: (labels, bboxes) produce a Tensor from torch.cat([labels, bboxes], dim=1)
    """
    layout: Tuple[str, ...]

    def convert(self, input: PostNMSDetections) -> Tensor:
        components = self.rearrange_components(input)
        return torch.cat(components, dim=1)


@dataclasses.dataclass
class TupleOfTensorsDetectionOutputFormat(DetectionOutputFormat):
    """
    Define the output format that return a tuple of tensors.
    A layout defines the order of tensors that is returned. For instance:
    - bboxes, scores, labels
    - labels, bboxes

    """
    layout: Tuple[str, ...]

    def convert(self, input: PostNMSDetections) -> Tuple[Tensor, ...]:
        components = self.rearrange_components(input)
        return tuple(components)


@dataclasses.dataclass
class DictDetectionOutputFormat(DetectionOutputFormat):
    """
    Define the output format that return a dictionary of tensors.
    A layout defines the key-value correspondence:
    - { bboxes: Tensor, scores: Tensor, labels: Tensor }
    - { labels: Tensor, bboxes: Tensor }
    """

    layout: Dict[str, str]

    def convert(self, input: PostNMSDetections) -> List[Dict[str, Tensor]]:
        components = self.rearrange_components(input)
        return dict(components)




@dataclasses.dataclass
class TensorSliceItem:
    location: slice
    name: str
    transform: Union[nn.Module, None] == nn.Identity()

    def get(self, input: Tensor) -> Tensor:
        return self.transform(input[..., self.location])

@dataclasses.dataclass
class BoundingBoxesTensorSliceItem(TensorSliceItem):
    format: BoundingBoxFormat

@dataclasses.dataclass
class ArgmaxTensorSliceItem(TensorSliceItem):
    def get(self, input: Tensor) -> Tensor:
        return torch.argmax(super().get(input), dim=-1, keepdim=True)

@dataclasses.dataclass
class MaxScoreTensorSliceItem(TensorSliceItem):
    def get(self, input: Tensor) -> Tensor:
        return torch.max(super().get(input), dim=-1, keepdim=True)


@dataclasses.dataclass
class ConcatenatedTensorPredictionsFormat:
    layout: Tuple[TensorSliceItem, ...]




class DetectionOutputAdapter(nn.Module):
    """
    Adapter class for converting detections for desired format

    Example:
    >>> class DetectX(nn.Module):
    >>>    ...
    >>>    @property
    >>>    def format(self):
    >>>        '''
    >>>        Describe the semantics of the model's output. In this example model's output consists of
    >>>         - Bounding boxes in XYXY format [4]
    >>>         - Predicted probas of N classes [N]
    >>>         - A distance predictions [1]
    >>>         - K additional labels [K]
    >>>        '''
    >>>        return ConcatenatedTensorPredictionsFormat(
    >>>            layout=(
    >>>                BoundingBoxesTensorSliceItem(location=slice(0, 4), name="bboxes", format=XYXYCoordinateFormat()),
    >>>                TensorSliceItem(location=slice(4, 4 + self.num_classes), name="scores"),
    >>>                TensorSliceItem(location=slice(4 + self.num_classes, 4 + self.num_classes + 1), name="distance")
    >>>                TensorSliceItem(location=slice(4 + self.num_classes + 1, 4 + self.num_classes + 1 + self.num_attributes), name="attributes")
    >>>            )
    >>>            # Alternatively one may use fluent builder interface to reduce chance of making an error specifing slice ranges:
    >>>            # layout=SlicedTensorBuilder.startsWith(BoundingBoxesTensorSliceItemFormat, 4, name="bboxes", format=XYXYCoordinateFormat()) \
    >>>            #                          .then(TensorSliceItem, self.num_classes, name="scores") \
    >>>            #                          .then(TensorSliceItem, 1, name="distance") \
    >>>            #                          .endsWith(TensorSliceItem, self.num_attributes, name="attributes")
    >>>        )
    >>>
    >>> yolox = MyCustomYolo(head=DetectX)
    >>>
    >>> # Suppose we want to return predictions in another format.
    >>> # Let it be:
    >>> # - Bounding boxes in normalized XYWH [4]
    >>> # - Predicted class label of most confident class [1]
    >>> # - Predicted probablity of the most confident class label [1]
    >>> # - Predicted attributes [K] with Sigmoid activation applied
    >>> # - Predicted distance [1] with ReLU applied to ensure non-negative output
    >>> output_format = ConcatenatedTensorPredictionsFormat(
    >>>     layout=(
    >>>         # Note source name refers here to a name of the item from the model's output format as specified in head
    >>>         BoundingBoxesTensorSliceItem(source="bboxes", format=NormalizedXYWHCoordinateFormat()),
    >>>         ArgmaxTensorSliceItem(source="scores"), # Compute class label
    >>>         MaxScoreTensorSliceItem(source="scores"), # Compute class score
    >>>         TensorSliceItem(source="attributes", transform=torch.nn.Sigmoid()),
    >>>         TensorSliceItem(source="distance", transform=torch.nn.ReLU())
    >>>     )
    >>> )
    >>>
    >>> # Now we can construct output adapter and attach it to the model
    >>> output_adapter = DetectionOutputAdapter(yolox,
    >>>     input_format=yolox.head.format,
    >>>     output_format=output_format
    >>> )
    >>>
    >>> yolox = nn.Sequential(yolox, output_adapter)
    >>>
    >>> # At some point we may return values as dictionary. What should happen then, is change of the format class to:
    >>> output_format = DictonaryPredictionsFormat(
    >>>     layout=(
    >>>         # Note source name refers here to a name of the item from the model's output format as specified in head
    >>>         BoundingBoxesTensorSliceItem(source="bboxes", format=NormalizedXYWHCoordinateFormat()),
    >>>         ArgmaxTensorSliceItem(source="scores"), # Compute class label
    >>>         MaxScoreTensorSliceItem(source="scores"), # Compute class score
    >>>         TensorSliceItem(source="attributes", transform=torch.nn.Sigmoid()),
    >>>         TensorSliceItem(source="distance", transform=torch.nn.ReLU())
    >>>     )
    >>> )
    """

    def __init__(self,
                 input_format: ConcatenatedTensorPredictionsFormat,
                 output_format: ConcatenatedTensorPredictionsFormat):
        super().__init__()
        self.input_format = input_format
        self.output_format = output_format

    def forward(self, predictions: Tensor):
        """
        Convert output detections to the user-specified format

        :param predictions:
        :return:
        """
        # TODO: Implement me
