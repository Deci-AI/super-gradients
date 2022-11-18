__all__ = ["DetectionOutputAdapter"]

from typing import Tuple, Union

from torch import nn, Tensor

from super_gradients.training.utils.output_adapters.formats import ConcatenatedTensorFormat


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

    def __init__(self, input_format: ConcatenatedTensorFormat, output_format: ConcatenatedTensorFormat, image_shape: Union[Tuple[int, int], None]):
        """

        :param input_format: Format definition of the inputs
        :param output_format: Format definition of the outputs
        :param image_shape: Shape of the input image (rows, cols), used for converting bbox coordinates from/to normalized formats. \
        If you're not using normalized coordinates you can set this to None
        """
        super().__init__()
        self.input_format = input_format
        self.output_format = output_format
        self.image_shape = image_shape

    def forward(self, predictions: Tensor, image_shape: Union[Tuple[int, int], None] = None) -> Tensor:
        """
        Convert output detections to the user-specified format
        :param predictions:
        :return:
        """
        if image_shape is None:
            image_shape = self.image_shape
        predictions = self.input_format.to_dict(predictions, image_shape=image_shape)
        return self.output_format.from_dict(predictions, image_shape=image_shape)
