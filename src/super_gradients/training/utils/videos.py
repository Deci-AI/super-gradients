from typing import List, Optional
import cv2
import os

import numpy as np

SUPPORTED_FORMATS = (".avi", ".mp4", ".mov", ".wmv", ".mkv")


def load_video_frames(file_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Open a video file and extract each frame into numpy array.

    :param file_path:   Path to the video file.
    :param max_frames:  Optional, maximum number of frames to extract.
    :return:            Frames, each as numpy arrays.
    """
    cap = _open_video(file_path)
    frames = _extract_frames(cap, max_frames)
    cap.release()  # Release the video capture object
    return frames


def _open_video(file_path: str) -> cv2.VideoCapture:
    """Open a video file.

    :param file_path:   Path to the video file
    :return:            Opened video capture object
    """
    ext = os.path.splitext(file_path)[-1].lower()  # TODO: Check this
    if ext not in SUPPORTED_FORMATS:
        raise RuntimeError(f"Not supported video format {ext}. Supported formats: {SUPPORTED_FORMATS}")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {file_path}")
    return cap


def _extract_frames(cap: cv2.VideoCapture, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Extract frames from an opened video capture object.

    :param cap:         Opened video capture object.
    :param max_frames:  Optional maximum number of frames to extract.
    :return:            Frames, each as numpy arrays.
    """
    frames = []

    while max_frames != len(frames):
        frame_read_success, frame = cap.read()
        if not frame_read_success:
            break
        frames.append(frame)

    return frames


# VIDEO_FILE_NAME = "test_video.mp4"
# video_frames = load_video_frames(VIDEO_FILE_NAME)

# print("Done")
# import cv2s
# import numpy as np
#
# # Define video settings
# VIDEO_FILE_NAME = "test_video.mp4"
# SOURCE = f"/Users/Louis.Dupont/PycharmProjects/super-gradients/src/super_gradients/training/utils/{VIDEO_FILE_NAME}"
# VIDEO_CODEC = cv2.VideoWriter_fourcc(*"mp4v")
# VIDEO_FPS = 30
# VIDEO_WIDTH = 640
# VIDEO_HEIGHT = 480
#
# # Create a VideoWriter object to write the frames to a video file
# video_writer = cv2.VideoWriter(VIDEO_FILE_NAME, VIDEO_CODEC, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
#
# # Define a color gradient to animate across the video frames
# gradient = np.linspace(0, 255, VIDEO_WIDTH, dtype=np.uint8)
#
# # Loop over the frames and draw a color gradient that moves across the screen
# for i in range(VIDEO_FPS * 5):
#     # Create a blank image with the background color
#     background_color = i % 255
#     frame = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), background_color, dtype=np.uint8)
#
#     # Calculate the current position of the gradient
#     gradient_pos = int((i / VIDEO_FPS) * VIDEO_WIDTH) % VIDEO_WIDTH
#
#     # Draw the color gradient on the frame
#     for x in range(VIDEO_WIDTH):
#         frame[:, x, :] = gradient[(gradient_pos + x) % VIDEO_WIDTH]
#
#     # Write the frame to the video file
#     video_writer.write(frame)
#
# # Release the VideoWriter object
# video_writer.release()
#
# print(f"Video saved to {SOURCE}")
