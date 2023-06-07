import os
from dataclasses import dataclass
from typing import List

import numpy as np

from super_gradients.training.utils.predict import ImagePrediction, ImagesPredictions, VideoPredictions, PoseEstimationPrediction
from super_gradients.training.utils.media.image import show_image, save_image
from super_gradients.training.utils.media.video import show_video_from_frames, save_video
from super_gradients.training.utils.visualization.poses import draw_skeleton


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
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw the predicted bboxes on the image.

        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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
        image = self.image.copy()

        for pred_i in np.argsort(self.prediction.scores):
            image = draw_skeleton(
                image=image,
                keypoints=self.prediction.poses[pred_i],
                score=self.prediction.scores[pred_i],
                show_confidence=show_confidence,
                joint_links=self.prediction.joint_links,
                joint_colors=joint_colors or self.prediction.joint_colors,
                joint_thickness=joint_thickness,
                keypoint_colors=keypoint_colors or self.prediction.keypoint_colors,
                keypoint_radius=keypoint_radius,
                box_thickness=box_thickness,
            )

        return image

    def show(
        self,
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> None:
        """Display the image with predicted bboxes.

        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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
            joint_colors=joint_colors,
            joint_thickness=joint_thickness,
            keypoint_colors=keypoint_colors,
            keypoint_radius=keypoint_radius,
            box_thickness=box_thickness,
            show_confidence=show_confidence,
        )
        show_image(image)

    def save(
        self,
        output_path: str,
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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


@dataclass
class ImagesPoseEstimationPrediction(ImagesPredictions):
    """Object wrapping the list of image detection predictions.

    :attr _images_prediction_lst:  List of the predictions results
    """

    _images_prediction_lst: List[ImagePoseEstimationPrediction]

    def show(
        self,
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> None:
        """Display the predicted bboxes on the images.

        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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
                joint_colors=joint_colors,
                joint_thickness=joint_thickness,
                keypoint_colors=keypoint_colors,
                keypoint_radius=keypoint_radius,
                box_thickness=box_thickness,
                show_confidence=show_confidence,
            )

    def save(
        self,
        output_folder: str,
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_folder:   Folder path, where the images will be saved.
        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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
                joint_colors=joint_colors,
                joint_thickness=joint_thickness,
                keypoint_colors=keypoint_colors,
                keypoint_radius=keypoint_radius,
                box_thickness=box_thickness,
                show_confidence=show_confidence,
            )


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
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> List[np.ndarray]:
        """Draw the predicted bboxes on the images.

        :param output_folder:   Folder path, where the images will be saved.
        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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
                joint_colors=joint_colors,
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
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> None:
        """Display the predicted bboxes on the images.

        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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
            joint_colors=joint_colors,
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
        joint_colors=None,
        joint_thickness: int = 2,
        keypoint_colors=None,
        keypoint_radius: int = 5,
        box_thickness: int = 2,
        show_confidence: bool = True,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param joint_colors:    Optional list of tuples representing the colors for each joint.
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
            joint_colors=joint_colors,
            joint_thickness=joint_thickness,
            keypoint_colors=keypoint_colors,
            keypoint_radius=keypoint_radius,
            box_thickness=box_thickness,
            show_confidence=show_confidence,
        )
        save_video(output_path=output_path, frames=frames, fps=self.fps)
