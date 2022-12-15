import copy
from typing import Tuple, Union, List

import numpy as np
import torch
from torch import nn, Tensor

from super_gradients.training.utils.bbox_formats import convert_bboxes
from super_gradients.training.utils.output_adapters.formats import ConcatenatedTensorFormat

__all__ = ["DetectionOutputAdapter"]


# class RearrangeNumpyArray:
#     """
#     Rearrange elements in last dimension of input tensor with respect to index argument
#
#     """
#
#     def __init__(self, indexes: np.ndarray):
#         self.indexes = indexes
#
#     def __call__(self, x: np.ndarray) -> np.ndarray:
#         """
#         :param x: Input tensor of  [..., N] shape
#         :return: Output tensor of [..., N[index]] shape
#         """
#         return x[..., self.indexes]


# class RearrangeTorchTensor(nn.Module):
#     """
#     Rearrange elements in last dimension of input tensor with respect to index argument
#
#     """
#
#     def __init__(self, indexes: Tensor):
#         super().__init__()
#         self.indexes = indexes
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         :param x: Input tensor of  [..., N] shape
#         :return: Output tensor of [..., N[index]] shape
#         """
#         if torch.jit.is_scripting():
#             # Workaround "Ellipses followed by tensor indexing is currently not supported"
#             # https://github.com/pytorch/pytorch/issues/34837
#             x = torch.moveaxis(x, -1, 0)
#             x = x[self.indexes]
#             x = torch.moveaxis(x, 0, -1)
#             return x
#         else:
#             return x[..., self.indexes]
#


def get_rearrange_outputs_module(input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat) -> Tuple[List, ConcatenatedTensorFormat]:
    # ) -> Tuple[Union[RearrangeNumpyArray, RearrangeTorchTensor], ConcatenatedTensorFormat]:

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
    return output_indexes, rearranged_format


def rearrange_tensor(tensor, indexes):
    if torch.jit.is_scripting() and isinstance(tensor, Tensor):
        # Workaround "Ellipses followed by tensor indexing is currently not supported"
        # https://github.com/pytorch/pytorch/issues/34837
        tensor = torch.moveaxis(tensor, -1, 0)
        tensor = tensor[indexes]
        tensor = torch.moveaxis(tensor, 0, -1)
        return tensor
    else:
        return tensor[..., indexes]


def convert_bboxes_format(tensor, source_format: ConcatenatedTensorFormat, target_format: ConcatenatedTensorFormat, image_shape):
    def _format_bboxes(bboxes):
        source_bboxes_format = source_format.bboxes_format.format
        target_bboxes_format = target_format.bboxes_format.format
        return convert_bboxes(
            source_format=source_bboxes_format,
            target_format=target_bboxes_format,
            bboxes=bboxes,
            inplace=False,
            image_shape=image_shape,
        )

    return source_format.apply_on_bbox(fn=_format_bboxes, concatenated_tensor=tensor)


class DetectionFormatAdapter:
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
        # self.rearrange_outputs, self.rearranged_format = self.get_rearrange_outputs_module(input_format, output_format)
        self.rearranged_indexes, self.rearranged_format = get_rearrange_outputs_module(input_format, output_format)

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
        predictions = rearrange_tensor(tensor=predictions, indexes=self.rearranged_indexes)
        predictions = convert_bboxes_format(tensor=predictions, source_format=self.input_format, target_format=self.output_format, image_shape=self.image_shape)
        return predictions

        # return self._get_rearrange_outputs_module_from_indexes(output_indexes), rearranged_format

    # @abstractmethod
    # def _get_rearrange_outputs_module_from_indexes(self, indexes: List) -> Union[RearrangeNumpyArray, RearrangeTorchTensor]:
    #     raise NotImplementedError


# class DetectionAdapterNumpy(DetectionAdapter):
#     def __init__(
#         self,
#         input_format: ConcatenatedTensorFormat,
#         output_format: ConcatenatedTensorFormat,
#         image_shape: Union[Tuple[int, int], None],
#     ):
#         """
#
#         :param input_format: Format definition of the inputs
#         :param output_format: Format definition of the outputs
#         :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
#                             If you're not using normalized coordinates you can set this to None
#         """
#         super().__init__(input_format=input_format, output_format=output_format, image_shape=image_shape)
#
#     def _get_rearrange_outputs_module_from_indexes(self, indexes: List) -> RearrangeNumpyArray:
#         return RearrangeNumpyArray(indexes=np.array(indexes).astype(np.long))
#
#
# class DetectionAdapterTorch(DetectionAdapter):
#     def __init__(
#         self,
#         input_format: ConcatenatedTensorFormat,
#         output_format: ConcatenatedTensorFormat,
#         image_shape: Union[Tuple[int, int], None],
#     ):
#         """
#
#         :param input_format: Format definition of the inputs
#         :param output_format: Format definition of the outputs
#         :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized format.
#                             If you're not using normalized coordinates you can set this to None
#         """
#         super().__init__(input_format=input_format, output_format=output_format, image_shape=image_shape)
#
#     def _get_rearrange_outputs_module_from_indexes(self, indexes: List) -> RearrangeTorchTensor:
#         return RearrangeTorchTensor(indexes=torch.tensor(indexes).long())


class DetectionOutputAdapter(nn.Module):
    """
    Adapter class for converting model's tensor for object detection to a desired format.
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
    >>>         - A distance tensor [1]
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
    >>> # Suppose we want to return tensor in another format.
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
        self.format_adapter = DetectionFormatAdapter(input_format=input_format, output_format=output_format, image_shape=image_shape)

    def forward(self, predictions: Tensor) -> Tensor:
        """
        Convert output detections to the user-specified format
        :param predictions:
        :return:
        """
        return self.format_adapter(predictions=predictions)
