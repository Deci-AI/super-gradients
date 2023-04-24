from typing import List, Optional, Tuple
import cv2
import PIL

import numpy as np


from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

__all__ = ["load_video", "save_video", "includes_video_extension", "show_video_from_disk", "show_video_from_frames"]

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".wmv", ".flv", ".gif")


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
    """Save a video locally. Depending on the extension, the video will be saved as a .mp4 file or as a .gif file.

    :param output_path: Where the video will be saved
    :param frames:      Frames representing the video, each in (H, W, C), RGB. Note that all the frames are expected to have the same shape.
    :param fps:         Frames per second
    """
    if not includes_video_extension(output_path):
        logger.info(f'Output path "{output_path}" does not have a video extension, and therefore will be saved as {output_path}.mp4')
        output_path += ".mp4"

    if check_is_gif(output_path):
        save_gif(output_path, frames, fps)
    else:
        save_mp4(output_path, frames, fps)


def save_gif(output_path: str, frames: List[np.ndarray], fps: int) -> None:
    """Save a video locally in .gif format.

    :param output_path: Where the video will be saved
    :param frames:      Frames representing the video, each in (H, W, C), RGB. Note that all the frames are expected to have the same shape.
    :param fps:         Frames per second
    """

    frames_pil = [PIL.Image.fromarray(frame) for frame in frames]

    frames_pil[0].save(output_path, save_all=True, append_images=frames_pil[1:], duration=int(1000 / fps), loop=0)


def save_mp4(output_path: str, frames: List[np.ndarray], fps: int) -> None:
    """Save a video locally in .mp4 format.

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


def show_video_from_disk(video_path: str, window_name: str = "Prediction"):
    """Display a video from disk using OpenCV.

    :param video_path:   Path to the video file.
    :param window_name:  Name of the window to display the video
    """
    cap = _open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Display the frame
            cv2.imshow(window_name, frame)

            # Wait for the specified number of milliseconds before displaying the next frame
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break
        else:
            break

    # Release the VideoCapture object and destroy the window
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def show_video_from_frames(frames: List[np.ndarray], fps: float, window_name: str = "Prediction") -> None:
    """Display a video from a list of frames using OpenCV.

    :param frames:      Frames representing the video, each in (H, W, C), RGB. Note that all the frames are expected to have the same shape.
    :param fps:         Frames per second
    :param window_name:  Name of the window to display the video
    """
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, frame)
        cv2.waitKey(int(1000 / fps))
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def includes_video_extension(file_path: str) -> bool:
    """Check if a file includes a video extension.
    :param file_path:   Path to the video file.
    :return:            True if the file includes a video extension.
    """
    return isinstance(file_path, str) and file_path.lower().endswith(VIDEO_EXTENSIONS)


def check_is_gif(file_path: str) -> bool:
    return file_path.lower().endswith(".gif")
