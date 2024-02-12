import sys

import cv2
import numpy as np
import time
from typing import Callable, Optional
from super_gradients.common.abstractions.abstract_logger import get_logger

__all__ = ["WebcamStreaming"]

logger = get_logger(__name__)


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
            message = "Could not open video capture device. Please check whether you have the webcam connected."
            if sys.platform == "darwin":
                message += " On macOS, you may need to grant the terminal access to the webcam in System Preferences."
                message += " Check https://stackoverflow.com/search?q=OpenCV+macOS+camera+access for more information."
            elif sys.platform == "nt":
                message += " On Windows, you may need to grant the terminal access to the webcam in the settings."
                message += " Check https://support.microsoft.com/en-us/windows/manage-app-permissions-for-your-camera-in-windows-87ebc757-1f87-7bbf-84b5-0686afb6ca6b#WindowsVersion=Windows_11 for more information."  # noqa
            raise ValueError(message)

        self._fps_counter = FPSCounter(update_frequency=fps_update_frequency)

    def run(self) -> None:
        """Start streaming video from the webcam and displaying it in a window.

        Press 'q' to quit the streaming.
        """
        while not self._stop() and self._display_single_frame():
            pass

    def _display_single_frame(self) -> bool:
        """Read a single frame from the video capture device, apply any specified frame processing,
        and display the resulting frame in the window.

        Also updates the FPS counter and displays it in the frame.
        """
        _ret, frame = self.cap.read()
        if not _ret or frame is None:
            logger.warning("Could not read frame from video capture device.")
            return False

        if self.frame_processing_fn:
            # Convert the frame to RGB since this is the format expected
            # by the predict function
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.frame_processing_fn(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        _write_fps_to_frame(frame, self.fps)
        cv2.imshow(self.window_name, frame)
        return _ret

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
