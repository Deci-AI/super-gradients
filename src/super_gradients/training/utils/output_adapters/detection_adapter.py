__all__ = ["DetectionOutputAdapter"]

import copy
from typing import Tuple, Union, Callable

import torch
from torch import nn, Tensor

from super_gradients.training.utils.bbox_formats import BoundingBoxFormat
from super_gradients.training.utils.output_adapters.formats import ConcatenatedTensorFormat, TensorSliceItem


class RearrangeOutput(nn.Module):
    """
    Rearrange elements in last dimension of input tensor with respect to index argument

    """

    def __init__(self, indexes: Tensor):
        super().__init__()
        self.indexes = indexes

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input tensor of  [..., N] shape
        :return: Output tensor of [..., N[index]] shape
        """
        if torch.jit.is_scripting():
            # Workaround "Ellipses followed by tensor indexing is currently not supported"
            # https://github.com/pytorch/pytorch/issues/34837
            x = torch.moveaxis(x, -1, 0)
            x = x[self.indexes]
            x = torch.moveaxis(x, 0, -1)
            return x
        else:
            return x[..., self.indexes]


class ConvertBoundingBoxes(nn.Module):
    to_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor]
    from_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor]

    def __init__(
        self,
        location: Tuple[int, int],
        to_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor],
        from_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor],
        image_shape: Tuple[int, int],
    ):
        super().__init__()
        self.to_xyxy = torch.jit.annotate(Callable[[Tensor, Tuple[int, int]], Tensor], to_xyxy)
        self.from_xyxy = torch.jit.annotate(Callable[[Tensor, Tuple[int, int]], Tensor], from_xyxy)
        self.image_shape = image_shape
        self.location = location

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x:
        :param image_shape:
        :return:
        """
        location = slice(self.location[0], self.location[1])
        bboxes = x[..., location]
        xyxy = self.to_xyxy(bboxes, self.image_shape)
        x[..., location] = self.from_xyxy(xyxy, self.image_shape)
        return x


class DetectionOutputAdapter(nn.Module):
    """
    Adapter class for converting model's predictions for object detection to a desired format:

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
    >>>         # Note: For output format it is not required to specify location attribute as it will be
    >>>         # computed with respect to size of "source name" and order of items in layout describe their order in the output tensor
    >>>         BoundingBoxesTensorSliceItem(source="bboxes", format=NormalizedXYWHCoordinateFormat()),
    >>>         TensorSliceItem(source="attributes"),
    >>>         TensorSliceItem(source="distance")
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
    >>>         TensorSliceItem(source="attributes"),
    >>>         TensorSliceItem(source="distance")
    >>>     )
    >>> )
    """

    def __init__(self, input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat, image_shape: Union[Tuple[int, int], None]):
        """

        :param input_format: Format definition of the inputs
        :param output_format: Format definition of the outputs
        :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
                            If you're not using normalized coordinates you can set this to None
        """
        super().__init__()
        self.rearrange_outputs, rearranged_format = self.get_rearrange_outputs_module(input_format, output_format)
        self.format_conversion: nn.Module = self.get_format_conversion_module(
            location=(rearranged_format.bboxes_format.location.start, rearranged_format.bboxes_format.location.stop),
            input_bbox_format=rearranged_format.bboxes_format.format,
            output_bbox_format=output_format.bboxes_format.format,
            image_shape=image_shape,
        )
        self.input_format = input_format
        self.output_format = output_format

    def forward(self, predictions: Tensor) -> Tensor:
        """
        Convert output detections to the user-specified format
        :param predictions:
        :return:
        """
        predictions = self.rearrange_outputs(predictions)
        predictions = self.format_conversion(predictions)
        return predictions

    @classmethod
    def get_rearrange_outputs_module(
        cls, input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat
    ) -> Tuple[RearrangeOutput, ConcatenatedTensorFormat]:

        output_indexes = []
        rearranged_layout = []

        offset = 0
        for output_name, output_spec in output_format.layout.items():
            if output_name not in input_format.layout:
                raise KeyError(f"Requested item '{output_name}' was not found among input format spec. Present items are: {tuple(input_format.layout.keys())}")

            input_element: TensorSliceItem = input_format.layout[output_name]
            indexes = list(range(input_element.location.start, input_element.location.stop))
            output_indexes.extend(indexes)
            output_len = len(indexes)

            rearranged_item = copy.deepcopy(output_spec)
            rearranged_item.location = slice(offset, offset + output_len)
            offset += output_len

            rearranged_layout.append(rearranged_item)
        rearranged_format = ConcatenatedTensorFormat(rearranged_layout)
        return RearrangeOutput(torch.tensor(output_indexes).long()), rearranged_format

    @classmethod
    def get_format_conversion_module(
        cls, location: Tuple[int, int], input_bbox_format: BoundingBoxFormat, output_bbox_format: BoundingBoxFormat, image_shape: Union[Tuple[int, int], None]
    ) -> ConvertBoundingBoxes:
        return ConvertBoundingBoxes(
            location=location,
            to_xyxy=input_bbox_format.get_to_xyxy(False),
            from_xyxy=output_bbox_format.get_from_xyxy(True),
            image_shape=image_shape,
        )
