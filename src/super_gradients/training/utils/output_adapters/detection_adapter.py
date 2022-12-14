import copy
from typing import Tuple, Union, Callable, List
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn, Tensor

from super_gradients.training.utils.bbox_formats import convert_bboxes
from super_gradients.training.utils.output_adapters.formats import ConcatenatedTensorFormat

__all__ = ["DetectionOutputAdapter"]


class RearrangeNumpyArray:
    """
    Rearrange elements in last dimension of input tensor with respect to index argument

    """

    def __init__(self, indexes: np.ndarray):
        self.indexes = indexes

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input tensor of  [..., N] shape
        :return: Output tensor of [..., N[index]] shape
        """
        return x[..., self.indexes]


class RearrangeTorchTensor(nn.Module):
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


class ConvertNumpyBoundingBoxes:
    def __init__(
        self,
        location: Tuple[int, int],
        to_xyxy: Callable[[np.ndarray, Tuple[int, int]], np.ndarray],
        from_xyxy: Callable[[np.ndarray, Tuple[int, int]], np.ndarray],
        image_shape: Tuple[int, int],
    ):
        pass  # TODO: implement

    def __call__(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        pass  # TODO: implement


# def apply_fn_to_layout(format: ConcatenatedTensorFormat, layout_name: str, function, concatenated_tensor):
#     location = format.locations[layout_name]
#     location = slice(location[0], location[1])
#     concatenated_tensor[..., location] = function(concatenated_tensor[..., location])
#     return concatenated_tensor
#
#
# def apply_fn_to_bbox(tensor_format: ConcatenatedTensorFormat, function: Callable, concatenated_tensor: Tensor):
#     """Apply the input function inplace"""
#     location = slice(tensor_format.bboxes_location[0], tensor_format.bboxes_location[1])
#     concatenated_tensor[..., location] = function(concatenated_tensor[..., location])
#     return concatenated_tensor
#
#
# def apply_to_location(function: Callable, tensor: Tensor, location: Tuple[int, int]):
#     location = slice(location[0], location[1])
#     tensor[..., location] = function(tensor[..., location])
#     return tensor


# class ConvertTorchBoundingBoxes(nn.Module):
#     to_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor]
#     from_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor]
#
#     def __init__(
#         self,
#         location: Tuple[int, int],
#         to_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor],
#         from_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor],
#         image_shape: Tuple[int, int],
#     ):
#         super().__init__()
#         self.to_xyxy = torch.jit.annotate(Callable[[Tensor, Tuple[int, int]], Tensor], to_xyxy)
#         self.from_xyxy = torch.jit.annotate(Callable[[Tensor, Tuple[int, int]], Tensor], from_xyxy)
#         self.image_shape = image_shape
#         self.location = location
#
#     def bbox_function(self, bboxes: Tensor) -> Tensor:
#         return self.from_xyxy(self.to_xyxy(bboxes, self.image_shape), self.image_shape)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#
#         :param x:
#         :param image_shape:
#         :return:
#         """
#         location = slice(self.location[0], self.location[1])
#         bboxes = x[..., location]
#         xyxy = self.to_xyxy(bboxes, self.image_shape)
#         x[..., location] = self.from_xyxy(xyxy, self.image_shape)
#         return x

# class ConvertNumpyBoundingBoxes(nn.Module):
#     to_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor]
#     from_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor]
#
#     def __init__(
#         self,
#         location: Tuple[int, int],
#         to_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor],
#         from_xyxy: Callable[[Tensor, Tuple[int, int]], Tensor],
#         image_shape: Tuple[int, int],
#     ):
#         super().__init__()
#         self.to_xyxy = Callable[[Tensor, Tuple[int, int]], Tensor], to_xyxy)
#         self.from_xyxy = Callable[[Tensor, Tuple[int, int]], Tensor], from_xyxy)
#         self.image_shape = image_shape
#         self.location = location
#
#     def bbox_function(self, bboxes: Tensor) -> Tensor:
#         return self.from_xyxy(self.to_xyxy(bboxes, self.image_shape), self.image_shape)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#
#         :param x:
#         :param image_shape:
#         :return:
#         """
#         location = slice(self.location[0], self.location[1])
#         bboxes = x[..., location]
#         xyxy = self.to_xyxy(bboxes, self.image_shape)
#         x[..., location] = self.from_xyxy(xyxy, self.image_shape)
#         return x


class DetectionAdapter(ABC):
    def __init__(
        self,
        input_format: ConcatenatedTensorFormat,
        output_format: ConcatenatedTensorFormat,
        image_shape: Union[Tuple[int, int], None],
    ):
        """

        :param input_format: Format definition of the inputs
        :param output_format: Format definition of the outputs
        :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
                            If you're not using normalized coordinates you can set this to None
        """
        super().__init__()
        self.rearrange_outputs, self.rearranged_format = self.get_rearrange_outputs_module(input_format, output_format)

        self.input_format = input_format
        self.output_format = output_format
        self.image_shape = image_shape
        self.input_length = input_format.num_channels

    def __call__(self, predictions: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        if predictions.shape[-1] != self.input_length:
            raise RuntimeError(
                f"Number of channels in last dimension of input tensor ({predictions.shape[-1]}) must be "
                f"equal to {self.input_length} as defined by input format."
            )
        predictions = self.rearrange_outputs(predictions)
        predictions = self.convert_bboxes_format(predictions=predictions, source_format=self.input_format, target_format=self.output_format)
        return predictions

    def get_rearrange_outputs_module(
        self, input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat
    ) -> Tuple[Union[RearrangeNumpyArray, RearrangeTorchTensor], ConcatenatedTensorFormat]:

        output_indexes = []
        rearranged_layout = []

        offset = 0
        for output_name, output_spec in output_format.layout.items():
            if output_name not in input_format.layout:
                raise KeyError(f"Requested item '{output_name}' was not found among input format spec. Present items are: {tuple(input_format.layout.keys())}")

            input_spec = input_format.layout[output_name]

            if input_spec.length != output_spec.length:
                raise RuntimeError(
                    "Length of the output must match in input and output format. "
                    "Input spec size is {input_spec.length} for key '{output_name}' and output spec size is {output_spec.length}."
                )
            indexes = input_format.indexes[output_name]
            output_indexes.extend(indexes)
            output_len = len(indexes)

            rearranged_item = copy.deepcopy(output_spec)
            offset += output_len

            rearranged_layout.append(rearranged_item)
        rearranged_format = ConcatenatedTensorFormat(rearranged_layout)
        return self._get_rearrange_outputs_module_from_indexes(output_indexes), rearranged_format

    @abstractmethod
    def _get_rearrange_outputs_module_from_indexes(self, indexes) -> Union[RearrangeNumpyArray, RearrangeTorchTensor]:
        raise NotImplementedError

    def convert_bboxes_format(self, predictions, source_format: ConcatenatedTensorFormat, target_format: ConcatenatedTensorFormat):
        def _format_bboxes(bboxes):
            source_bboxes_format = source_format.bboxes_format.format
            target_bboxes_format = target_format.bboxes_format.format
            return convert_bboxes(
                source_format=source_bboxes_format,
                target_format=target_bboxes_format,
                bboxes=bboxes,
                inplace=False,
                image_shape=self.image_shape,
            )

        return source_format.apply_on_bbox(fn=_format_bboxes, concatenated_tensor=predictions)


class DetectionAdapterNumpy(DetectionAdapter):
    def __init__(
        self,
        input_format: ConcatenatedTensorFormat,
        output_format: ConcatenatedTensorFormat,
        image_shape: Union[Tuple[int, int], None],
    ):
        """

        :param input_format: Format definition of the inputs
        :param output_format: Format definition of the outputs
        :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
                            If you're not using normalized coordinates you can set this to None
        """
        super().__init__(input_format=input_format, output_format=output_format, image_shape=image_shape)

    def _get_rearrange_outputs_module_from_indexes(self, indexes: List) -> RearrangeNumpyArray:
        return RearrangeNumpyArray(indexes=np.array(indexes).astype(np.long))


class DetectionAdapterTorch(DetectionAdapter):
    def __init__(
        self,
        input_format: ConcatenatedTensorFormat,
        output_format: ConcatenatedTensorFormat,
        image_shape: Union[Tuple[int, int], None],
    ):
        """

        :param input_format: Format definition of the inputs
        :param output_format: Format definition of the outputs
        :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
                            If you're not using normalized coordinates you can set this to None
        """
        super().__init__(input_format=input_format, output_format=output_format, image_shape=image_shape)

    def _get_rearrange_outputs_module_from_indexes(self, indexes: List) -> RearrangeTorchTensor:
        return RearrangeTorchTensor(indexes=torch.tensor(indexes).long())


class DetectionOutputAdapter(nn.Module):
    """
    Adapter class for converting model's predictions for object detection to a desired format.
    This adapter supports torch.jit tracing & scripting & onnx conversion.

    >>> from super_gradients.training.utils.output_adapters.formats import ConcatenatedTensorFormat, BoundingBoxesTensorSliceItem, TensorSliceItem
    >>> from super_gradients.training.utils.bbox_formats import XYXYCoordinateFormat, NormalizedXYWHCoordinateFormat
    >>>
    >>> class CustomDetectionHead(nn.Module):
    >>>    num_classes: int = 123
    >>>
    >>>    @property
    >>>    def format(self):
    >>>        '''
    >>>        Describe the semantics of the model's output. In this example model's output consists of
    >>>         - Bounding boxes in XYXY format [4]
    >>>         - Predicted probas of N classes [N]
    >>>         - A distance predictions [1]
    >>>         - K additional labels [K]
    >>>        '''
    >>>        return ConcatenatedTensorFormat(
    >>>            layout=(
    >>>                BoundingBoxesTensorSliceItem(name="bboxes", format=XYXYCoordinateFormat()),
    >>>                TensorSliceItem(name="label", length=1),
    >>>                TensorSliceItem(name="distance", length=1),
    >>>                TensorSliceItem(name="attributes", length=4),
    >>>            )
    >>>        )
    >>>
    >>> yolox = YoloX(head=CustomDetectionHead)
    >>>
    >>> # Suppose we want to return predictions in another format.
    >>> # Let it be:
    >>> # - Bounding boxes in normalized XYWH [4]
    >>> # - Predicted attributes [4]
    >>> # - Predicted label [1]
    >>> output_format = ConcatenatedTensorFormat(
    >>>     layout=(
    >>>         # Note: For output format it is not required to specify location attribute as it will be
    >>>         # computed with respect to size of "source name" and order of items in layout describe their order in the output tensor
    >>>         BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedXYWHCoordinateFormat()),
    >>>         TensorSliceItem(name="attributes", length=4),
    >>>         TensorSliceItem(name="label", length=1),
    >>>     )
    >>> )
    >>>
    >>> # Now we can construct output adapter and attach it to the model
    >>> output_adapter = DetectionOutputAdapter(yolox,
    >>>     input_format=yolox.head.format,
    >>>     output_format=output_format,
    >>>     image_shape=(640, 640)
    >>> )
    >>>
    >>> yolox = nn.Sequential(yolox, output_adapter)
    >>>
    """

    def __init__(self, input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat, image_shape: Union[Tuple[int, int], None]):
        """

        :param input_format: Format definition of the inputs
        :param output_format: Format definition of the outputs
        :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
                            If you're not using normalized coordinates you can set this to None
        """
        super().__init__()
        self.adapter = DetectionAdapterTorch(input_format=input_format, output_format=output_format, image_shape=image_shape)

    def forward(self, predictions: Tensor) -> Tensor:
        """
        Convert output detections to the user-specified format
        :param predictions:
        :return:
        """
        return self.adapter(predictions=predictions)
