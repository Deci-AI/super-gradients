from typing import List, Optional, Tuple
import cv2

import numpy as np


__all__ = ["load_video", "save_video"]


def load_video(file_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], int]:
    """Open a video file and extract each frame into numpy array.

    :param file_path:   Path to the video file.
    :param max_frames:  Optional, maximum number of frames to extract.
    :return:
                - Frames representing the video, each in (H, W, C), RGB.
                - Frames per Second (FPS).
    """
    cap = _open_video(file_path)
    frames = _extract_frames(cap, max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def _open_video(file_path: str) -> cv2.VideoCapture:
    """Open a video file.

    :param file_path:   Path to the video file
    :return:            Opened video capture object
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {file_path}")
    return cap


def _extract_frames(cap: cv2.VideoCapture, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Extract frames from an opened video capture object.

    :param cap:         Opened video capture object.
    :param max_frames:  Optional maximum number of frames to extract.
    :return:            Frames representing the video, each in (H, W, C), RGB.
    """
    frames = []

    while max_frames != len(frames):
        frame_read_success, frame = cap.read()
        if not frame_read_success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return frames


def save_video(output_path: str, frames: List[np.ndarray], fps: int) -> None:
    """Save a video locally.

    :param output_path: Where the video will be saved
    :param frames:      Frames representing the video, each in (H, W, C), RGB. Note that all the frames are expected to have the same shape.
    :param fps:         Frames per second
    """
    video_height, video_width = _validate_frames(frames)

    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_width, video_height),
    )

    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()


def _validate_frames(frames: List[np.ndarray]) -> Tuple[float, float]:
    """Validate the frames to make sure that every frame has the same size and includes the channel dimension. (i.e. (H, W, C))

    :param frames:  Frames representing the video, each in (H, W, C), RGB. Note that all the frames are expected to have the same shape.
    :return:        (Height, Weight) of the video.
    """
    min_height = min(frame.shape[0] for frame in frames)
    max_height = max(frame.shape[0] for frame in frames)

    min_width = min(frame.shape[1] for frame in frames)
    max_width = max(frame.shape[1] for frame in frames)

    if (min_height, min_width) != (max_height, max_width):
        raise RuntimeError(
            f"Your video is made of frames that have (height, width) going from ({min_height}, {min_width}) to ({max_height}, {max_width}).\n"
            f"Please make sure that all the frames have the same shape."
        )

    if set(frame.ndim for frame in frames) != {3} or set(frame.shape[-1] for frame in frames) != {3}:
        raise RuntimeError("Your frames must include 3 channels.")

    return max_height, max_width


# print("")
# print("Done")
# import cv2
# import numpy as np
#
# # Define video settings
# VIDEO_FILE_NAME = "/home/louis.dupont/PycharmProjects/test_video.mp4"
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
#         frame[:, :, 0] = 200
#     # Write the frame to the video file
#     video_writer.write(frame)
#
# # Release the VideoWriter object
# video_writer.release()
#
# print(f"Video saved to {VIDEO_FILE_NAME}")
