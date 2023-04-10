import cv2
import numpy as np
import time
from typing import Callable, Optional


class Streaming:
    """Stream video from a webcam."""

    def __init__(self, frame_processing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None, capture: int = cv2.CAP_ANY):
        self.frame_processing_fn = frame_processing_fn
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

        self.cap = cv2.VideoCapture(capture)
        print(capture)

    def run(self):
        while self._run_single_frame():
            continue
        self.cap.release()
        cv2.destroyAllWindows()

    def _run_single_frame(self):

        # read a frame from the webcam
        ret, frame = self.cap.read()

        # pass the image through the `predict` function to draw bounding boxes
        if self.frame_processing_fn:
            frame = self.frame_processing_fn(frame)
        self._update_fps()

        # add FPS text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)
        line_type = 2
        cv2.putText(frame, "FPS: {:.2f}".format(self.fps), (10, 30), font, font_scale, font_color, line_type)

        # display the image with bounding boxes drawn and FPS in the top left corner
        cv2.imshow("frame", frame)

        # wait for a key press to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        return True

    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.start_time = current_time
            self.frame_count = 0
