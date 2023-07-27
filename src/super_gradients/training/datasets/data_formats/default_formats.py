from super_gradients.common.object_names import ConcatenatedTensorFormats
from super_gradients.training.datasets.data_formats.formats import ConcatenatedTensorFormat, BoundingBoxesTensorSliceItem, LabelTensorSliceItem
from super_gradients.training.datasets.data_formats.bbox_formats import (
    XYXYCoordinateFormat,
    XYWHCoordinateFormat,
    CXCYWHCoordinateFormat,
    NormalizedXYXYCoordinateFormat,
    NormalizedXYWHCoordinateFormat,
    NormalizedCXCYWHCoordinateFormat,
)

XYXY_LABEL = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=XYXYCoordinateFormat()),
        LabelTensorSliceItem(),
    )
)
XYWH_LABEL = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=XYWHCoordinateFormat()),
        LabelTensorSliceItem(),
    )
)
CXCYWH_LABEL = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=CXCYWHCoordinateFormat()),
        LabelTensorSliceItem(),
    )
)
LABEL_XYXY = ConcatenatedTensorFormat(
    layout=(
        LabelTensorSliceItem(),
        BoundingBoxesTensorSliceItem(name="bboxes", format=XYXYCoordinateFormat()),
    )
)
LABEL_XYWH = ConcatenatedTensorFormat(
    layout=(
        LabelTensorSliceItem(),
        BoundingBoxesTensorSliceItem(name="bboxes", format=XYWHCoordinateFormat()),
    )
)
LABEL_CXCYWH = ConcatenatedTensorFormat(
    layout=(
        LabelTensorSliceItem(),
        BoundingBoxesTensorSliceItem(name="bboxes", format=CXCYWHCoordinateFormat()),
    )
)
NORMALIZED_XYXY_LABEL = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedXYXYCoordinateFormat()),
        LabelTensorSliceItem(),
    )
)
NORMALIZED_XYWH_LABEL = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedXYWHCoordinateFormat()),
        LabelTensorSliceItem(),
    )
)
NORMALIZED_CXCYWH_LABEL = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedCXCYWHCoordinateFormat()),
        LabelTensorSliceItem(),
    )
)
LABEL_NORMALIZED_XYXY = ConcatenatedTensorFormat(
    layout=(
        LabelTensorSliceItem(),
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedXYXYCoordinateFormat()),
    )
)
LABEL_NORMALIZED_XYWH = ConcatenatedTensorFormat(
    layout=(
        LabelTensorSliceItem(),
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedXYWHCoordinateFormat()),
    )
)
LABEL_NORMALIZED_CXCYWH = ConcatenatedTensorFormat(
    layout=(
        LabelTensorSliceItem(),
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedCXCYWHCoordinateFormat()),
    )
)


def get_default_data_format(format_name: str) -> ConcatenatedTensorFormat:
    return DEFAULT_CONCATENATED_TENSOR_FORMATS[format_name]


DEFAULT_CONCATENATED_TENSOR_FORMATS = {
    ConcatenatedTensorFormats.XYXY_LABEL: XYXY_LABEL,
    ConcatenatedTensorFormats.XYWH_LABEL: XYWH_LABEL,
    ConcatenatedTensorFormats.CXCYWH_LABEL: CXCYWH_LABEL,
    ConcatenatedTensorFormats.LABEL_XYXY: LABEL_XYXY,
    ConcatenatedTensorFormats.LABEL_XYWH: LABEL_XYWH,
    ConcatenatedTensorFormats.LABEL_CXCYWH: LABEL_CXCYWH,
    ConcatenatedTensorFormats.NORMALIZED_XYXY_LABEL: NORMALIZED_XYXY_LABEL,
    ConcatenatedTensorFormats.NORMALIZED_XYWH_LABEL: NORMALIZED_XYWH_LABEL,
    ConcatenatedTensorFormats.NORMALIZED_CXCYWH_LABEL: NORMALIZED_CXCYWH_LABEL,
    ConcatenatedTensorFormats.LABEL_NORMALIZED_XYXY: LABEL_NORMALIZED_XYXY,
    ConcatenatedTensorFormats.LABEL_NORMALIZED_XYWH: LABEL_NORMALIZED_XYWH,
    ConcatenatedTensorFormats.LABEL_NORMALIZED_CXCYWH: LABEL_NORMALIZED_CXCYWH,
}
