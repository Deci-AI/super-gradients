import cv2
import numpy as np


def cxcywhr_to_poly(boxes: np.ndarray) -> np.ndarray:
    """
    Convert oriented bounding boxes in CX-CY-W-H-R format to a polygon format
    :param boxes: [N,...,5] oriented bounding boxes in CX-CY-W-H-R format
    :return: [N,...,4, 2] oriented bounding boxes in polygon format
    """
    shape = boxes.shape
    if shape[-1] != 5:
        raise ValueError(f"Expected last dimension to be 5, got {shape[-1]}")

    flat_rboxes = boxes.reshape(-1, 5)
    polys = np.zeros((flat_rboxes.shape[0], 4, 2), dtype=np.float32)
    for i, box in enumerate(flat_rboxes):
        cx, cy, w, h, r = box
        rect = ((cx, cy), (w, h), np.rad2deg(r))
        poly = cv2.boxPoints(rect)
        polys[i] = poly

    return polys.reshape(*shape[:-1], 4, 2)


def poly_to_cxcywhr(poly: np.ndarray) -> np.ndarray:
    shape = poly.shape
    if shape[-2:] != (4, 2):
        raise ValueError(f"Expected last two dimensions to be (4, 2), got {shape[-2:]}")

    flat_polys = poly.reshape(-1, 4, 2)
    rboxes = np.zeros((flat_polys.shape[0], 5), dtype=np.float32)
    for i, poly in enumerate(flat_polys):
        hull = cv2.convexHull(np.reshape(poly, [-1, 2]))
        rect = cv2.minAreaRect(hull)
        cx, cy = rect[0]
        w, h = rect[1]
        angle = rect[2]
        if angle == 0:
            w, h = h, w
            angle -= 90
        rboxes[i] = [cx, cy, w, h, angle]

    return rboxes.reshape(*shape[:-2], 5)
