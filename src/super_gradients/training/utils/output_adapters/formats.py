import collections
from typing import Tuple, Union, List, Mapping

from super_gradients.training.utils.bbox_formats import BoundingBoxFormat


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
