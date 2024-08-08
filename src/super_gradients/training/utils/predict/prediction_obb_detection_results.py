import os

import numpy as np
import cv2

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterator, Iterable, Union

from super_gradients.training.utils.media.image import save_image, show_image
from super_gradients.training.utils.media.video import show_video_from_frames, save_video
from super_gradients.training.utils.visualization.obb import OBBVisualization
from super_gradients.training.utils.visualization.utils import generate_color_mapping
from tqdm import tqdm

from .predictions import Prediction
from .prediction_results import ImagePrediction, VideoPredictions, ImagesPredictions

__all__ = ["OBBDetectionPrediction", "ImageOBBDetectionPrediction", "ImagesOBBDetectionPrediction", "VideoOBBDetectionPrediction"]


@dataclass
class OBBDetectionPrediction(Prediction):
    """Represents an OBB detection prediction, with bboxes represented in cxycxwhr format."""

    rboxes_cxcywhr: np.ndarray
    confidence: np.ndarray
    labels: np.ndarray

    def __init__(self, rboxes_cxcywhr: np.ndarray, confidence: np.ndarray, labels: np.ndarray, image_shape: Tuple[int, int]):
        """
        :param rboxes_cxcywhr: Rboxes of [N,5] shape in the CXCYWHR format
        :param confidence:  Confidence scores for each bounding box
        :param labels:      Labels for each bounding box.
        :param image_shape: Shape of the image the prediction is made on, (H, W). This is used to convert bboxes to xyxy format
        """
        self._validate_input(rboxes_cxcywhr, confidence, labels)
        self.rboxes_cxcywhr = rboxes_cxcywhr
        self.confidence = confidence
        self.labels = labels
        self.image_shape = image_shape

    def _validate_input(self, rboxes_cxcywhr: np.ndarray, confidence: np.ndarray, labels: np.ndarray) -> None:
        n_bboxes, n_confidences, n_labels = rboxes_cxcywhr.shape[0], confidence.shape[0], labels.shape[0]
        if n_bboxes != n_confidences != n_labels:
            raise ValueError(
                f"The number of bounding boxes ({n_bboxes}) does not match the number of confidence scores ({n_confidences}) and labels ({n_labels})."
            )
        if rboxes_cxcywhr.shape[1] != 5:
            raise ValueError(f"Expected 5 columns in rboxes_cxcywhr, got {rboxes_cxcywhr.shape[1]}.")

    def __len__(self):
        return len(self.rboxes_cxcywhr)


@dataclass
class ImageOBBDetectionPrediction(ImagePrediction):
    """Object wrapping an image and a detection model's prediction.

    :param image:        Input image
    :param prediction:   Predictions of the model
    :param class_names:  List of the class names to predict
    """

    image: np.ndarray
    prediction: OBBDetectionPrediction
    class_names: List[str]

    def draw(
        self,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_rboxes: Optional[np.ndarray] = None,
        target_class_ids: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Draw the predicted bboxes on the image.

        :param box_thickness:           (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence:         Whether to show confidence scores on the image.
        :param color_mapping:           List of tuples representing the colors for each class.
                                        Default is None, which generates a default color mapping based on the number of class names.
        :param target_rboxes:           Optional[Union[np.ndarray, List[np.ndarray]]], ground truth bounding boxes.
                                        Can either be an np.ndarray of shape (image_i_object_count, 4) when predicting a single image,
                                        or a list of length len(target_bboxes), containing such arrays.
                                        When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one)
        :param target_class_ids:        Optional[Union[np.ndarray, List[np.ndarray]]], ground truth target class indices. Can either be an np.ndarray of shape
                                        (image_i_object_count) when predicting a single image, or a list of length len(target_bboxes), containing such arrays.
        :param target_bboxes_format:    Optional[str], bounding box format of target_bboxes, one of
                                        ['xyxy','xywh', 'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh'].
                                        Will raise an error if not None and target_bboxes is None.
        :param class_names:             List of class names to show. By default, is None which shows all classes using during training.

        :return:                Image with predicted bboxes. Note that this does not modify the original image.
        """
        target_rboxes = target_rboxes if target_rboxes is not None else np.zeros((0, 5))
        target_class_ids = target_class_ids if target_class_ids is not None else np.zeros((0, 1))

        class_names_to_show = class_names if class_names else self.class_names
        class_ids_to_show = [i for i, class_name in enumerate(self.class_names) if class_name in class_names_to_show]
        invalid_class_names_to_show = set(class_names_to_show) - set(self.class_names)
        if len(invalid_class_names_to_show) > 0:
            raise ValueError(
                "`class_names` includes class names that the model was not trained on.\n"
                f"    - Invalid class names:   {list(invalid_class_names_to_show)}\n"
                f"    - Available class names: {list(self.class_names)}"
            )

        plot_targets = target_rboxes is not None and len(target_rboxes)
        color_mapping = color_mapping or generate_color_mapping(len(class_names_to_show))

        keep_mask = np.isin(self.prediction.labels, class_ids_to_show)
        image = OBBVisualization.draw_obb(
            image=self.image.copy(),
            rboxes_cxcywhr=self.prediction.rboxes_cxcywhr[keep_mask],
            scores=self.prediction.confidence[keep_mask],
            labels=self.prediction.labels[keep_mask],
            class_names=class_names_to_show,
            class_colors=color_mapping,
            show_labels=True,
            show_confidence=show_confidence,
            thickness=box_thickness,
        )

        if plot_targets:
            keep_mask = np.isin(target_class_ids, class_ids_to_show)
            target_image = OBBVisualization.draw_obb(
                image=self.image.copy(),
                rboxes_cxcywhr=target_rboxes[keep_mask],
                scores=None,
                labels=target_class_ids[keep_mask],
                class_names=class_names_to_show,
                class_colors=color_mapping,
                show_labels=True,
                show_confidence=False,
                thickness=box_thickness,
            )

            height, width, ch = target_image.shape
            new_width, new_height = int(width + width / 20), int(height + height / 8)

            # Crate a new canvas with new width and height.
            canvas_image = np.ones((new_height, new_width, ch), dtype=np.uint8) * 255
            canvas_target = np.ones((new_height, new_width, ch), dtype=np.uint8) * 255

            # New replace the center of canvas with original image
            padding_top, padding_left = 60, 10

            canvas_image[padding_top : padding_top + height, padding_left : padding_left + width] = image
            canvas_target[padding_top : padding_top + height, padding_left : padding_left + width] = target_image

            img1 = cv2.putText(canvas_image, "Predictions", (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            img2 = cv2.putText(canvas_target, "Ground Truth", (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

            image = cv2.hconcat((img1, img2))
        return image

    def show(
        self,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[np.ndarray] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Display the image with predicted bboxes.

        :param box_thickness:           (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence:         Whether to show confidence scores on the image.
        :param color_mapping:           List of tuples representing the colors for each class.
                                        Default is None, which generates a default color mapping based on the number of class names.
        :param target_bboxes:           Optional[Union[np.ndarray, List[np.ndarray]]], ground truth bounding boxes.
                                        Can either be an np.ndarray of shape (image_i_object_count, 4) when predicting a single image,
                                        or a list of length len(target_bboxes), containing such arrays.
                                        When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one)
        :param target_class_ids:        Optional[Union[np.ndarray, List[np.ndarray]]], ground truth target class indices. Can either be an np.ndarray of shape
                                        (image_i_object_count) when predicting a single image, or a list of length len(target_bboxes), containing such arrays.
        :param target_bboxes_format:    Optional[str], bounding box format of target_bboxes, one of
                                        ['xyxy','xywh', 'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh'].
                                        Will raise an error if not None and target_bboxes is None.
        :param class_names:             List of class names to show. By default, is None which shows all classes using during training.
        """
        image = self.draw(
            box_thickness=box_thickness,
            show_confidence=show_confidence,
            color_mapping=color_mapping,
            target_rboxes=target_bboxes,
            target_class_ids=target_class_ids,
            class_names=class_names,
        )
        show_image(image)

    def save(
        self,
        output_path: str,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[np.ndarray] = None,
        target_class_ids: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:             Path to the output video file.
        :param box_thickness:           (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence:         Whether to show confidence scores on the image.
        :param color_mapping:           List of tuples representing the colors for each class.
                                        Default is None, which generates a default color mapping based on the number of class names.
        :param target_bboxes:           Optional[Union[np.ndarray, List[np.ndarray]]], ground truth bounding boxes.
                                        Can either be an np.ndarray of shape (image_i_object_count, 4) when predicting a single image,
                                        or a list of length len(target_bboxes), containing such arrays.
                                        When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one)
        :param target_class_ids:        Optional[Union[np.ndarray, List[np.ndarray]]], ground truth target class indices. Can either be an np.ndarray of shape
                                        (image_i_object_count) when predicting a single image, or a list of length len(target_bboxes), containing such arrays.
        :param class_names:             List of class names to show. By default, is None which shows all classes using during training.
        """
        image = self.draw(
            box_thickness=box_thickness,
            show_confidence=show_confidence,
            color_mapping=color_mapping,
            target_rboxes=target_bboxes,
            target_class_ids=target_class_ids,
            class_names=class_names,
        )
        save_image(image=image, path=output_path)


@dataclass
class ImagesOBBDetectionPrediction(ImagesPredictions):
    """Object wrapping the list of image detection predictions.

    :attr _images_prediction_lst:  List of the predictions results
    """

    _images_prediction_lst: List[ImageOBBDetectionPrediction]

    def show(
        self,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Display the predicted bboxes on the images.

        :param box_thickness:           (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence:         Whether to show confidence scores on the image.
        :param color_mapping:           List of tuples representing the colors for each class.
                                        Default is None, which generates a default color mapping based on the number of class names.
        :param target_bboxes:           Optional[Union[np.ndarray, List[np.ndarray]]], ground truth bounding boxes.
                                        Can either be an np.ndarray of shape (image_i_object_count, 4) when predicting a single image,
                                        or a list of length len(target_bboxes), containing such arrays.
                                        When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one)
        :param target_class_ids:        Optional[Union[np.ndarray, List[np.ndarray]]], ground truth target class indices. Can either be an np.ndarray of shape
                                        (image_i_object_count) when predicting a single image, or a list of length len(target_bboxes), containing such arrays.
        :param target_bboxes_format:    Optional[str], bounding box format of target_bboxes, one of
                                        ['xyxy','xywh', 'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh'].
                                        Will raise an error if not None and target_bboxes is None.
        :param class_names:             List of class names to show. By default, is None which shows all classes using during training.
        """
        target_bboxes, target_class_ids = self._check_target_args(target_bboxes, target_bboxes_format, target_class_ids)

        for prediction, target_bbox, target_class_id in zip(self._images_prediction_lst, target_bboxes, target_class_ids):
            prediction.show(
                box_thickness=box_thickness,
                show_confidence=show_confidence,
                color_mapping=color_mapping,
                target_bboxes=target_bbox,
                target_bboxes_format=target_bboxes_format,
                target_class_ids=target_class_id,
                class_names=class_names,
            )

    def _check_target_args(
        self,
        target_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ):
        if not (
            (target_bboxes is None and target_bboxes_format is None and target_class_ids is None)
            or (target_bboxes is not None and target_bboxes_format is not None and target_class_ids is not None)
        ):
            raise ValueError("target_bboxes, target_bboxes_format, and target_class_ids should either all be None or all not None.")

        if isinstance(target_bboxes, np.ndarray):
            target_bboxes = [target_bboxes]
        if isinstance(target_class_ids, np.ndarray):
            target_class_ids = [target_class_ids]

        if target_bboxes is not None and target_class_ids is not None and len(target_bboxes) != len(target_class_ids):
            raise ValueError(f"target_bboxes and target_class_ids lengths should be equal, got: {len(target_bboxes)} and {len(target_class_ids)}.")
        if target_bboxes is not None and target_class_ids is not None and len(target_bboxes) != len(self._images_prediction_lst):
            raise ValueError(
                f"target_bboxes and target_class_ids lengths should be equal, to the "
                f"amount of images passed to predict(), got: {len(target_bboxes)} and {len(self._images_prediction_lst)}."
            )
        if target_bboxes is None:
            target_bboxes = [None for _ in range(len(self._images_prediction_lst))]
            target_class_ids = [None for _ in range(len(self._images_prediction_lst))]

        return target_bboxes, target_class_ids

    def save(
        self,
        output_folder: str,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_folder:           Folder path, where the images will be saved.
        :param box_thickness:           (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence:         Whether to show confidence scores on the image.
        :param color_mapping:           List of tuples representing the colors for each class.
                                        Default is None, which generates a default color mapping based on the number of class names.
        :param target_bboxes:           Optional[Union[np.ndarray, List[np.ndarray]]], ground truth bounding boxes.
                                        Can either be an np.ndarray of shape (image_i_object_count, 4) when predicting a single image,
                                        or a list of length len(target_bboxes), containing such arrays.
                                        When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one)
        :param target_class_ids:        Optional[Union[np.ndarray, List[np.ndarray]]], ground truth target class indices. Can either be an np.ndarray of shape
                                        (image_i_object_count) when predicting a single image, or a list of length len(target_bboxes), containing such arrays.
        :param target_bboxes_format:    Optional[str], bounding box format of target_bboxes, one of
                                        ['xyxy','xywh', 'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh'].
                                        Will raise an error if not None and target_bboxes is None.
        :param class_names:             List of class names to show. By default, is None which shows all classes using during training.
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        target_bboxes, target_class_ids = self._check_target_args(target_bboxes, target_bboxes_format, target_class_ids)

        for i, (prediction, target_bbox, target_class_id) in enumerate(zip(self._images_prediction_lst, target_bboxes, target_class_ids)):
            image_output_path = os.path.join(output_folder, f"pred_{i}.jpg")
            prediction.save(
                output_path=image_output_path,
                box_thickness=box_thickness,
                show_confidence=show_confidence,
                color_mapping=color_mapping,
                class_names=class_names,
            )


@dataclass
class VideoOBBDetectionPrediction(VideoPredictions):
    """Object wrapping the list of image detection predictions as a Video.

    :attr _images_prediction_gen:   Iterable object of the predictions results
    :att fps:                       Frames per second of the video
    """

    _images_prediction_gen: Iterable[ImageOBBDetectionPrediction]
    fps: int
    n_frames: int

    def draw(
        self,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> Iterator[np.ndarray]:
        """Draw the predicted bboxes on the images.

        :param box_thickness:   (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :param class_names:     List of class names to show. By default, is None which shows all classes using during training.
        :return:                Iterable object of images with predicted bboxes. Note that this does not modify the original image.
        """

        for result in tqdm(self._images_prediction_gen, total=self.n_frames, desc="Processing Video"):
            yield result.draw(
                box_thickness=box_thickness,
                show_confidence=show_confidence,
                color_mapping=color_mapping,
                class_names=class_names,
            )

    def show(
        self,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Display the predicted bboxes on the images.

        :param box_thickness:   (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :param class_names:     List of class names to show. By default, is None which shows all classes using during training.
        """
        frames = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping, class_names=class_names)
        show_video_from_frames(window_name="Detection", frames=frames, fps=self.fps)

    def save(
        self,
        output_path: str,
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param box_thickness:   (Optional) Thickness of bounding boxes. If None, will adapt to the box size.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :param class_names:     List of class names to show. By default, is None which shows all classes using during training.
        """
        frames = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping, class_names=class_names)
        save_video(output_path=output_path, frames=frames, fps=self.fps)
