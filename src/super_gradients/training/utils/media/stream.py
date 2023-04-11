import cv2
import numpy as np
import time
from typing import Callable, Optional


__all__ = ["WebcamStreaming"]


class WebcamStreaming:
    """Stream video from a webcam. Press 'q' to quit the streaming.

    :param window_name:          Name of the window to display the video stream.
    :param frame_processing_fn:  Function to apply to each frame before displaying it.
                                 If None, frames are displayed as is.
    :param capture:              ID of the video capture device to use.
                                 Default is cv2.CAP_ANY (which selects the first available device).
    :param fps_update_frequency: Minimum time (in seconds) between updates to the FPS counter.
                                 If None, the counter is updated every frame.
    """

    def __init__(
        self,
        window_name: str = "",
        frame_processing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        capture: int = cv2.CAP_ANY,
        fps_update_frequency: Optional[float] = None,
    ):
        self.window_name = window_name
        self.frame_processing_fn = frame_processing_fn
        self.cap = cv2.VideoCapture(capture)
        if not self.cap.isOpened():
            raise ValueError("Could not open video capture device")

        self._fps_counter = FPSCounter(update_frequency=fps_update_frequency)

    def run(self) -> None:
        """Start streaming video from the webcam and displaying it in a window.

        Press 'q' to quit the streaming.
        """
        while not self._stop():
            self._display_single_frame()

    def _display_single_frame(self) -> None:
        """Read a single frame from the video capture device, apply any specified frame processing,
        and display the resulting frame in the window.

        Also updates the FPS counter and displays it in the frame.
        """
        _ret, frame = self.cap.read()

        if self.frame_processing_fn:
            frame = self.frame_processing_fn(frame)

        _write_fps_to_frame(frame, self.fps)
        cv2.imshow(self.window_name, frame)

    def _stop(self) -> bool:
        """Stopping condition for the streaming."""
        return cv2.waitKey(1) & 0xFF == ord("q")

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    def __del__(self):
        """Release the video capture device and close the window."""
        self.cap.release()
        cv2.destroyAllWindows()


def _write_fps_to_frame(frame: np.ndarray, fps: float) -> None:
    """Write the current FPS value on the given frame.

    :param frame:   Frame to write the FPS value on.
    :param fps:     Current FPS value to write.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)
    line_type = 2
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30), font, font_scale, font_color, line_type)


class FPSCounter:
    """Class for calculating the FPS of a video stream."""

    def __init__(self, update_frequency: Optional[float] = None):
        """Create a new FPSCounter object.

        :param update_frequency: Minimum time (in seconds) between updates to the FPS counter.
                                 If None, the counter is updated every frame.
        """
        self._update_frequency = update_frequency

        self._start_time = time.time()
        self._frame_count = 0
        self._fps = 0.0

    def _update_fps(self, elapsed_time, current_time) -> None:
        """Compute new value of FPS and reset the counter."""
        self._fps = self._frame_count / elapsed_time
        self._start_time = current_time
        self._frame_count = 0

    @property
    def fps(self) -> float:
        """Current FPS value."""

        self._frame_count += 1
        current_time, elapsed_time = time.time(), time.time() - self._start_time

        if self._update_frequency is None or elapsed_time > self._update_frequency:
            self._update_fps(elapsed_time=elapsed_time, current_time=current_time)

        return self._fps
