from typing import Optional, Union, List, Tuple

import cv2
import numpy as np


class OBBVisualization:
    @classmethod
    def draw_obb(
        self,
        image: np.ndarray,
        rboxes_cxcywhr: np.ndarray,
        scores: Optional[np.ndarray],
        labels: np.ndarray,
        class_labels,
        class_colors: Union[List[Tuple], np.ndarray],
        show_labels: bool = True,
        show_confidence: bool = True,
        thickness: int = 2,
        opacity: float = 0.75,
        label_prefix: str = "",
    ):
        """
        Draw rotated bounding boxes on the image

        :param image: [H, W, 3] - Image to draw bounding boxes on
        :param rboxes_cxcywhr: [N, 5] - List of rotated bounding boxes in format [cx, cy, w, h, r]
        :param labels: [N] - List of class indices
        :param scores: [N] - List of confidence scores. Can be None, in which case confidence is not shown
        :param class_labels: [C] - List of class names
        :param class_colors: [C, 3] - List of class colors
        :param thickness: Thickness of the bounding box
        :param show_labels: Boolean flag that indicates if labels should be shown (Default: True)
        :param show_confidence: Boolean flag that indicates if confidence should be shown (Default: True)
        :param opacity: Opacity of the overlay (Default: 0.5)
        :param label_prefix: Prefix for the label (Default: "")

        :return: [H, W, 3] - Image with bounding boxes drawn
        """
        if len(class_labels) != len(class_colors):
            raise ValueError("Number of class labels and colors should match")

        overlay = image.copy()
        num_boxes = len(rboxes_cxcywhr)

        font_face = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.0

        show_confidence = show_confidence and scores is not None

        if scores is not None:
            # Reorder the boxes to start with boxes of the lowest confidence
            order = np.argsort(scores)
            rboxes_cxcywhr = rboxes_cxcywhr[order]
            scores = scores[order]
            labels = labels[order]

        for i in range(num_boxes):
            cx, cy, w, h, r = rboxes_cxcywhr[i]
            rect = (cx, cy), (w, h), np.rad2deg(r)
            box = cv2.boxPoints(rect)  # [4, 2]
            class_index = int(labels[i])
            color = tuple(class_colors[class_index])
            cv2.polylines(overlay, box[None, :, :].astype(int), True, color, thickness=thickness, lineType=cv2.LINE_AA)

            if show_labels:
                class_label = class_labels[class_index]
                label_title = f"{label_prefix}{class_label}"
                if show_confidence:
                    conf = scores[i]
                    label_title = f"{label_title} {conf:.2f}"

                text_size, centerline = cv2.getTextSize(label_title, font_face, font_scale, thickness)
                #  Place origin somewhere at the top/top-right corner, use top-right corner of the `box`
                org = (int(box[1][0]), int(box[1][1] - text_size[1]))
                cv2.putText(overlay, label_title, org=org, fontFace=font_face, fontScale=font_scale, color=color, lineType=cv2.LINE_AA)

        return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
