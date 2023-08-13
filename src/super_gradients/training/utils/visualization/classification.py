import cv2
import numpy as np


def draw_label(image: np.ndarray, label: str, confidence: float) -> np.ndarray:
    """Draw a label and confidence on an image.
    :param image:       The image on which to draw the label and confidence, in RGB format, and Channel Last (H, W, C)
    :param label:       The label to draw.
    :param confidence:  The confidence of the label.
    """

    # Format confidence as a percentage
    confidence_str = f"{confidence * 100:.3f}%"

    # Use a slightly smaller font scale and a moderate thickness
    fontScale = 0.8
    thickness = 1

    # Define additional spacing between the two lines
    line_spacing = 5

    # Determine the size of the label and confidence text
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]
    confidence_size = cv2.getTextSize(confidence_str, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]

    # Determine the size of the bounding rectangle
    text_width = max(label_size[0], confidence_size[0])
    text_height = label_size[1] + confidence_size[1] + thickness * 3 + line_spacing

    # Calculate the position to draw the label, centered horizontally and at the top
    start_x = (image.shape[1] - text_width) // 2
    start_y = 5

    # Draw a filled rectangle with transparency as the background for the label
    overlay = image.copy()
    bg_color = (255, 255, 255)  # White
    bg_start = (start_x, start_y)
    bg_end = (start_x + text_width, start_y + text_height)
    cv2.rectangle(overlay, bg_start, bg_end, bg_color, thickness=-1)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Center the label and confidence text within the bounding rectangle, with additional spacing
    text_color = (0, 0, 0)  # Black
    cv2.putText(
        image,
        label,
        (start_x + (text_width - label_size[0]) // 2, start_y + label_size[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        confidence_str,
        (start_x + (text_width - confidence_size[0]) // 2, start_y + label_size[1] + confidence_size[1] + thickness + line_spacing),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    return image
