import collections
from typing import Tuple, Union, List

from torch import Tensor

from super_gradients.training.utils.bbox_formats import BoundingBoxFormat, convert_bboxes


class DetectionOutputFormat:
    pass


class TensorSliceItem:
    location: slice
    name: str

    def __init__(self, name: str, location: slice):
        self.name = name
        self.location = location

    def __repr__(self):
        return f"name={self.name} location={self.location}"


class BoundingBoxesTensorSliceItem(TensorSliceItem):
    format: BoundingBoxFormat

    def __init__(self, name: str, location: slice, format: BoundingBoxFormat):
        super().__init__(name, location)
        self.format = format

    def get_output(self, values: Tensor, output_format: "BoundingBoxesTensorSliceItem", **kwargs):
        image_shape = kwargs.get("image_shape", None)
        return convert_bboxes(values, image_shape=image_shape, source_format=output_format.format, target_format=self.format, inplace=False)

    def __repr__(self):
        return f"name={self.name} location={self.location} format={self.format}"


class ConcatenatedTensorFormat(DetectionOutputFormat):
    """
    Define the output format that return a single tensor of shape [N,M] (N - number of detections,
    M - sum of bbox attributes) that is a concatenated from bbox coordinates and other fields.
    A layout defines the order of concatenated tensors. For instance:
    - layout: (bboxes, scores, labels) gives a Tensor that is product of torch.cat([bboxes, scores, labels], dim=1)
    - layout: (labels, bboxes) produce a Tensor from torch.cat([labels, bboxes], dim=1)
    """

    layout: collections.OrderedDict[str, TensorSliceItem]

    @property
    def bboxes_format(self) -> BoundingBoxesTensorSliceItem:
        bbox_items = [x for x in self.layout.values() if isinstance(x, BoundingBoxesTensorSliceItem)]
        return bbox_items[0]

    def __init__(self, layout: Union[List[TensorSliceItem], Tuple[TensorSliceItem, ...]]):
        bbox_items = [x for x in layout if isinstance(x, BoundingBoxesTensorSliceItem)]
        if len(bbox_items) != 1:
            raise RuntimeError("Number of bounding box items must be strictly equal to 1")

        self.layout = collections.OrderedDict([(item.name, item) for item in layout])
