import os
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from super_gradients.training.utils.predict import ImagePrediction, ImagesPredictions, VideoPredictions, PoseEstimationPrediction
from super_gradients.training.utils.media.image import show_image, save_image
from super_gradients.training.utils.media.video import show_video_from_frames, save_video
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization


@dataclass
class ImagePoseEstimationPrediction(ImagePrediction):
    """Object wrapping an image and a detection model's prediction.

    :attr image:        Input image
    :attr predictions:  Predictions of the model
    :attr class_names:  List of the class names to predict
    """

    image: np.ndarray
    prediction: PoseEstimationPrediction

    def draw(
        self,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
        font_size="auto",
    ) -> np.ndarray:
        """Draw the predicted bboxes on the image.

        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.
        :return:                Image with predicted bboxes. Note that this does not modify the original image.
        """
        min_size = max(self.image.shape[:2])

        if box_thickness == "auto":
            box_thickness = min(10, max(1, int(min_size / 300)))
        else:
            box_thickness = int(box_thickness)

        if joint_thickness == "auto":
            joint_thickness = min(10, max(1, int(min_size / 300)))
        else:
            joint_thickness = int(joint_thickness)

        if keypoint_radius == "auto":
            keypoint_radius = 2 * joint_thickness

        if font_size == "auto":
            font_size = min_size / 800

        image = PoseVisualization.draw_poses(
            image=self.image,
            poses=self.prediction.poses,
            scores=self.prediction.scores,
            is_crowd=None,
            boxes=self.prediction.bboxes_xyxy,
            edge_links=self.prediction.edge_links,
            edge_colors=edge_colors or self.prediction.edge_colors,
            joint_thickness=joint_thickness,
            keypoint_colors=keypoint_colors or self.prediction.keypoint_colors,
            keypoint_radius=keypoint_radius,
            box_thickness=box_thickness,
            font_size=font_size,
        )

        return image

    def show(
        self,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
    ) -> "ImagePoseEstimationPrediction":
        """Display the image with predicted bboxes.

        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.
        """
        image = self.draw(
            edge_colors=edge_colors,
            joint_thickness=joint_thickness,
            keypoint_colors=keypoint_colors,
            keypoint_radius=keypoint_radius,
            box_thickness=box_thickness,
            show_confidence=show_confidence,
        )
        show_image(image)
        return self

    def save(
        self,
        output_path: str,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
    ) -> "ImagePoseEstimationPrediction":
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.
        """
        image = self.draw(box_thickness=box_thickness, show_confidence=show_confidence)
        save_image(image=image, path=output_path)
        return self


@dataclass
class ImagesPoseEstimationPrediction(ImagesPredictions):
    """Object wrapping the list of image detection predictions.

    :attr _images_prediction_lst:  List of the predictions results
    """

    _images_prediction_lst: List[ImagePoseEstimationPrediction]

    def show(
        self,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
    ) -> "ImagesPoseEstimationPrediction":
        """Display the predicted bboxes on the images.

        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.
        """
        for prediction in self._images_prediction_lst:
            prediction.show(
                edge_colors=edge_colors,
                joint_thickness=joint_thickness,
                keypoint_colors=keypoint_colors,
                keypoint_radius=keypoint_radius,
                box_thickness=box_thickness,
                show_confidence=show_confidence,
            )
        return self

    def save(
        self,
        output_folder: str,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_folder:   Folder path, where the images will be saved.
        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        for i, prediction in enumerate(self._images_prediction_lst):
            image_output_path = os.path.join(output_folder, f"pred_{i}.jpg")
            prediction.save(
                output_path=image_output_path,
                edge_colors=edge_colors,
                joint_thickness=joint_thickness,
                keypoint_colors=keypoint_colors,
                keypoint_radius=keypoint_radius,
                box_thickness=box_thickness,
                show_confidence=show_confidence,
            )
        return self


@dataclass
class VideoPoseEstimationPrediction(VideoPredictions):
    """Object wrapping the list of image detection predictions as a Video.

    :attr _images_prediction_lst:   List of the predictions results
    :att fps:                       Frames per second of the video
    """

    _images_prediction_lst: List[ImagePoseEstimationPrediction]
    fps: int

    def draw(
        self,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
    ) -> List[np.ndarray]:
        """Draw the predicted bboxes on the images.

        :param output_folder:   Folder path, where the images will be saved.
        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.

        :return:                List of images with predicted bboxes. Note that this does not modify the original image.
        """
        frames_with_bbox = [
            result.draw(
                edge_colors=edge_colors,
                joint_thickness=joint_thickness,
                keypoint_colors=keypoint_colors,
                keypoint_radius=keypoint_radius,
                box_thickness=box_thickness,
                show_confidence=show_confidence,
            )
            for result in self._images_prediction_lst
        ]
        return frames_with_bbox

    def show(
        self,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
    ) -> None:
        """Display the predicted bboxes on the images.

        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.
        """
        frames = self.draw(
            edge_colors=edge_colors,
            joint_thickness=joint_thickness,
            keypoint_colors=keypoint_colors,
            keypoint_radius=keypoint_radius,
            box_thickness=box_thickness,
            show_confidence=show_confidence,
        )
        show_video_from_frames(window_name="Pose Estimation", frames=frames, fps=self.fps)

    def save(
        self,
        output_path: str,
        edge_colors=None,
        joint_thickness: Union[int, str] = "auto",
        keypoint_colors=None,
        keypoint_radius: Union[int, str] = "auto",
        box_thickness: Union[int, str] = "auto",
        show_confidence: bool = False,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param edge_colors:    Optional list of tuples representing the colors for each joint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joint links in the skeleton.
        :param joint_thickness: Thickness of the joint links  (in pixels).
        :param keypoint_colors: Optional list of tuples representing the colors for each keypoint.
                                If None, default colors are used.
                                If not None the length must be equal to the number of joints in the skeleton.
        :param keypoint_radius: Radius of the keypoints (in pixels).
        :param show_confidence: Whether to show confidence scores on the image.
        :param box_thickness:   Thickness of bounding boxes.
        """
        frames = self.draw(
            edge_colors=edge_colors,
            joint_thickness=joint_thickness,
            keypoint_colors=keypoint_colors,
            keypoint_radius=keypoint_radius,
            box_thickness=box_thickness,
            show_confidence=show_confidence,
        )
        save_video(output_path=output_path, frames=frames, fps=self.fps)
