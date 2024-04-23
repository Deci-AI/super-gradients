import cv2
import numpy as np


class OBBVisualization:
    @classmethod
    def draw_obb(
        self,
        image: np.ndarray,
        rboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        class_labels,
        classs_colors: np.ndarray,
        show_labels: bool = True,
        show_confidence: bool = True,
        thickness=2,
        opacity=0.5,
        label_prefix="",
    ):
        """


        :param image:
        :param boxes:
        :param labels:
        :param class_labels: [C]
        :param classs_colors: [C]
        :param thickness:
        :param show_labels:
        :param opacity:

        Returns:

        """
        overlay = image.copy()
        num_boxes = len(rboxes)

        font_face = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.0

        # Reorder the boxes to start with boxes of the lowest confidence
        order = np.argsort(scores)
        rboxes = rboxes[order]
        scores = scores[order]
        labels = labels[order]

        for i in range(num_boxes):
            cx, cy, w, h, r = rboxes[i]
            rect = (cx, cy), (w, h), r
            box = cv2.boxPoints(rect)
            class_index = labels[i]
            color = tuple(classs_colors[class_index])
            cv2.polylines(overlay, box, True, color, thickness=thickness, lineType=cv2.LINE_AA)

            if show_labels:
                class_label = class_labels[class_index]
                label_title = f"{label_prefix}{class_label}"
                if show_confidence:
                    conf = scores[class_index]
                    label_title = f"{label_title} {conf:.2f}"

                text_size, centerline = cv2.getTextSize(label_title, font_face, font_scale, thickness)
                org = (int(cx), int(cy))  # TODO: Place origin somewhere at the top/top-right corner
                cv2.putText(overlay, label_title, org=org, fontFace=font_face, fontScale=font_scale, color=color, lineType=cv2.LINE_AA)

        return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
