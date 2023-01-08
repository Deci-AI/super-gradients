from .bbox_format import BoundingBoxFormat, convert_bboxes
from .cxcywh import CXCYWHCoordinateFormat
from .xywh import XYWHCoordinateFormat
from .xyxy import XYXYCoordinateFormat
from .yxyx import YXYXCoordinateFormat
from .normalized_xyxy import NormalizedXYXYCoordinateFormat
from .normalized_cxcywh import NormalizedCXCYWHCoordinateFormat
from .normalized_xywh import NormalizedXYWHCoordinateFormat

BBOX_FORMATS = {
    "xyxy": XYXYCoordinateFormat,
    "xywh": XYWHCoordinateFormat,
    "yxyx": YXYXCoordinateFormat,
    "cxcywh": CXCYWHCoordinateFormat,
    "normalized_xyxy": NormalizedXYXYCoordinateFormat,
    "normalized_xywh": NormalizedXYWHCoordinateFormat,
    "normalized_cxcywh": NormalizedCXCYWHCoordinateFormat,
}

__all__ = [
    "BBOX_FORMATS",
    "BoundingBoxFormat",
    "CXCYWHCoordinateFormat",
    "NormalizedCXCYWHCoordinateFormat",
    "NormalizedXYWHCoordinateFormat",
    "NormalizedXYXYCoordinateFormat",
    "XYWHCoordinateFormat",
    "XYXYCoordinateFormat",
    "YXYXCoordinateFormat",
    "convert_bboxes",
]
