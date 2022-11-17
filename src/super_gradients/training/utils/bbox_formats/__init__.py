from .bbox_format import BoundingBoxFormat, convert_bboxes
from .cxcywh import CXCYWHCoordinateFormat, NormalizedCXCYWHCoordinateFormat, cxcywh2xyxy, xyxy2cxcywh
from .xywh import XYWHCoordinateFormat, NormalizedXYWHCoordinateFormat
from .xyxy import XYXYCoordinateFormat, NormalizedXYXYCoordinateFormat
from .yxyx import YXYXCoordinateFormat


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
    "cxcywh2xyxy",
    "xyxy2cxcywh",
]
