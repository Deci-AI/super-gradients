import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterator, Union

import cv2
import numpy as np

from super_gradients.common.factories.bbox_format_factory import BBoxFormatFactory
from super_gradients.training.utils.media.image import show_image, save_image
from super_gradients.training.utils.media.video import show_video_from_frames, save_video
from super_gradients.training.utils.visualization.detection import draw_bbox
from super_gradients.training.utils.visualization.classification import draw_label

from super_gradients.training.utils.visualization.utils import generate_color_mapping
from .predictions import Prediction, DetectionPrediction, ClassificationPrediction
from ...datasets.data_formats.bbox_formats import convert_bboxes


@dataclass
class ImagePrediction(ABC):
    """Object wrapping an image and a model's prediction.

    :attr image:        Input image
    :attr predictions:  Predictions of the model
    :attr class_names:  List of the class names to predict
    """

    image: np.ndarray
    prediction: Prediction
    class_names: List[str]

    @abstractmethod
    def draw(self, *args, **kwargs) -> np.ndarray:
        """Draw the predictions on the image."""
        pass

    @abstractmethod
    def show(self, *args, **kwargs) -> None:
        """Display the predictions on the image."""
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        """Save the predictions on the image."""
        pass


@dataclass
class ImageClassificationPrediction(ImagePrediction):
    """Object wrapping an image and a classification model's prediction.

    :attr image:        Input image
    :attr predictions:  Predictions of the model
    :attr class_names:  List of the class names to predict
    """

    image: np.ndarray
    prediction: ClassificationPrediction
    class_names: List[str]

    def draw(self, show_confidence: bool = True) -> np.ndarray:
        """Draw the predicted label on the image.

        :param show_confidence: Whether to show confidence scores on the image.
        :return:                Image with predicted label.
        """

        image = self.image.copy()
        return draw_label(image=image, label=self.class_names[self.prediction.label], confidence=self.prediction.confidence)

    def show(self, show_confidence: bool = True) -> None:
        """Display the image with predicted label.

        :param show_confidence: Whether to show confidence scores on the image.
        """
        # to do draw the prediction on the image
        image = self.draw(show_confidence=show_confidence)
        show_image(image)

    def save(
        self,
        output_path: str,
        show_confidence: bool = True,
    ) -> None:
        """Save the predicted label on the images.

        :param output_path:     Path to the output video file.
        :param show_confidence: Whether to show confidence scores on the image.
        """
        image = self.draw(show_confidence=show_confidence)
        save_image(image=image, path=output_path)


@dataclass
class ImageDetectionPrediction(ImagePrediction):
    """Object wrapping an image and a detection model's prediction.

    :attr image:        Input image
    :attr predictions:  Predictions of the model
    :attr class_names:  List of the class names to predict
    """

    image: np.ndarray
    prediction: DetectionPrediction
    class_names: List[str]

    def draw(
        self,
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[np.ndarray] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw the predicted bboxes on the image.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.

        :param target_bboxes: Optional[np.ndarray], ground truth bounding boxes represented as an np.ndarray of shape
        (image_i_object_count, 4). When not None, will plot the predictions and the ground truth bounding boxes side
        by side (i.e 2 images stitched as one). (default=None).

        :param target_class_ids: Optional[np.ndarray], ground truth target class indices
        represented as an np.ndarray of shape (object_count). (default=None).

        :param target_bboxes_format: Optional[str], bounding box format of target_bboxes, one of ['xyxy','xywh',
        'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh']. Ignored if not
        None and target_bboxes is None.

        :return:                Image with predicted bboxes. Note that this does not modify the original image.

        """
        image = self.image.copy()

        target_bboxes = target_bboxes if target_bboxes is not None else np.zeros((0, 4))
        target_class_ids = target_class_ids if target_class_ids is not None else np.zeros((0, 1))
        bbox_format_factory = BBoxFormatFactory()
        if len(target_bboxes):
            target_bboxes_xyxy = convert_bboxes(
                bboxes=target_bboxes,
                image_shape=self.prediction.image_shape,
                source_format=bbox_format_factory.get(target_bboxes_format),
                target_format=bbox_format_factory.get("xyxy"),
                inplace=False,
            )
        else:
            target_bboxes_xyxy = target_bboxes

        plot_targets = any([len(tbbx) > 0 for tbbx in target_bboxes_xyxy])
        color_mapping = color_mapping or generate_color_mapping(len(self.class_names))

        for pred_i in np.argsort(self.prediction.confidence):
            class_id = int(self.prediction.labels[pred_i])
            score = "" if not show_confidence else str(round(self.prediction.confidence[pred_i], 2))
            image = draw_bbox(
                image=image,
                title=f"{self.class_names[class_id]} {score}",
                color=color_mapping[class_id],
                box_thickness=box_thickness,
                x1=int(self.prediction.bboxes_xyxy[pred_i, 0]),
                y1=int(self.prediction.bboxes_xyxy[pred_i, 1]),
                x2=int(self.prediction.bboxes_xyxy[pred_i, 2]),
                y2=int(self.prediction.bboxes_xyxy[pred_i, 3]),
            )

        if plot_targets:
            target_image = self.image.copy()
            for target_idx in range(len(target_bboxes_xyxy)):
                class_id = int(target_class_ids[target_idx])
                target_image = draw_bbox(
                    image=target_image,
                    title=f"{self.class_names[class_id]}",
                    color=color_mapping[class_id],
                    box_thickness=box_thickness,
                    x1=int(target_bboxes_xyxy[target_idx, 0]),
                    y1=int(target_bboxes_xyxy[target_idx, 1]),
                    x2=int(target_bboxes_xyxy[target_idx, 2]),
                    y2=int(target_bboxes_xyxy[target_idx, 3]),
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
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[np.ndarray] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[np.ndarray] = None,
    ) -> None:

        """Display the image with predicted bboxes.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.

        :param target_bboxes: Optional[np.ndarray], ground truth bounding boxes represented as an np.ndarray of shape
        (image_i_object_count, 4). When not None, will plot the predictions and the ground truth bounding boxes side
        by side (i.e 2 images stitched as one). (default=None).

        :param target_class_ids: Optional[np.ndarray], ground truth target class indices
        represented as an np.ndarray of shape (object_count). (default=None).

        :param target_bboxes_format: Optional[str], bounding box format of target_bboxes, one of ['xyxy','xywh',
        'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh']. Ignored if not
        None and target_bboxes is None.
        """
        image = self.draw(
            box_thickness=box_thickness,
            show_confidence=show_confidence,
            color_mapping=color_mapping,
            target_bboxes=target_bboxes,
            target_bboxes_format=target_bboxes_format,
            target_class_ids=target_class_ids,
        )
        show_image(image)

    def save(
        self,
        output_path: str,
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[np.ndarray] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.

        :param target_bboxes: Optional[np.ndarray], ground truth bounding boxes represented as an np.ndarray of shape
        (image_i_object_count, 4). When not None, will plot the predictions and the ground truth bounding boxes side
        by side (i.e 2 images stitched as one). (default=None).

        :param target_class_ids: Optional[np.ndarray], ground truth target class indices
        represented as an np.ndarray of shape (object_count). (default=None).

        :param target_bboxes_format: Optional[str], bounding box format of target_bboxes, one of ['xyxy','xywh',
        'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh']. Ignored if not
        None and target_bboxes is None.
        """
        image = self.draw(
            box_thickness=box_thickness,
            show_confidence=show_confidence,
            color_mapping=color_mapping,
            target_bboxes=target_bboxes,
            target_bboxes_format=target_bboxes_format,
            target_class_ids=target_class_ids,
        )
        save_image(image=image, path=output_path)


@dataclass
class ImagesPredictions(ABC):
    """Object wrapping the list of image predictions.

    :attr _images_prediction_lst: List of results of the run
    """

    _images_prediction_lst: List[ImagePrediction]

    def __len__(self) -> int:
        return len(self._images_prediction_lst)

    def __getitem__(self, index: int) -> ImagePrediction:
        return self._images_prediction_lst[index]

    def __iter__(self) -> Iterator[ImagePrediction]:
        return iter(self._images_prediction_lst)

    @abstractmethod
    def show(self, *args, **kwargs) -> None:
        """Display the predictions on the images."""
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        """Save the predictions on the images."""
        pass


@dataclass
class VideoPredictions(ImagesPredictions, ABC):
    """Object wrapping the list of image predictions as a Video.

    :attr _images_prediction_lst:   List of results of the run
    :att fps:                       Frames per second of the video
    """

    _images_prediction_lst: List[ImagePrediction]
    fps: float

    @abstractmethod
    def show(self, *args, **kwargs) -> None:
        """Display the predictions on the video."""
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        """Save the predictions on the video."""
        pass


@dataclass
class ImagesClassificationPrediction(ImagesPredictions):
    """Object wrapping the list of image classification predictions.

    :attr _images_prediction_lst:  List of the predictions results
    """

    _images_prediction_lst: List[ImageClassificationPrediction]

    def show(self, show_confidence: bool = True) -> None:
        """Display the predicted labels on the images.
        :param show_confidence: Whether to show confidence scores on the image.
        """
        for prediction in self._images_prediction_lst:
            prediction.show(show_confidence=show_confidence)

    def save(self, output_folder: str, show_confidence: bool = True) -> None:
        """Save the predicted label on the images.

        :param output_folder:     Folder path, where the images will be saved.
        :param show_confidence: Whether to show confidence scores on the image.
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        for i, prediction in enumerate(self._images_prediction_lst):
            image_output_path = os.path.join(output_folder, f"pred_{i}.jpg")
            prediction.save(output_path=image_output_path, show_confidence=show_confidence)


@dataclass
class ImagesDetectionPrediction(ImagesPredictions):
    """Object wrapping the list of image detection predictions.

    :attr _images_prediction_lst:  List of the predictions results
    """

    _images_prediction_lst: List[ImageDetectionPrediction]

    def show(
        self,
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> None:
        """Display the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :param target_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]], ground truth bounding boxes. Can either be an np.ndarray of shape
         (image_i_object_count, 4) when predicting a single image, or a list of length len(target_bboxes), containing such arrays.
         When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one).

        :param target_class_ids: Optional[Union[np.ndarray, List[np.ndarray]]], ground truth target class indices. Can either be an np.ndarray of shape
         (image_i_object_count) when predicting a single image, or a list of length len(target_bboxes), containing such arrays (default=None).

        :param target_bboxes_format: Optional[str], bounding box format of target_bboxes, one of ['xyxy','xywh',
        'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh']. Will raise an
        error if not None and target_bboxes is None.
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
        box_thickness: int = 2,
        show_confidence: bool = True,
        color_mapping: Optional[List[Tuple[int, int, int]]] = None,
        target_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        target_bboxes_format: Optional[str] = None,
        target_class_ids: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_folder:     Folder path, where the images will be saved.
        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :param target_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]], ground truth bounding boxes. Can either be an np.ndarray of shape
         (image_i_object_count, 4) when predicting a single image, or a list of length len(target_bboxes), containing such arrays.
         When not None, will plot the predictions and the ground truth bounding boxes side by side (i.e 2 images stitched as one).

        :param target_class_ids: Optional[Union[np.ndarray, List[np.ndarray]]], ground truth target class indices. Can either be an np.ndarray of shape
         (image_i_object_count) when predicting a single image, or a list of length len(target_bboxes), containing such arrays (default=None).

        :param target_bboxes_format: Optional[str], bounding box format of target_bboxes, one of ['xyxy','xywh',
        'yxyx' 'cxcywh' 'normalized_xyxy' 'normalized_xywh', 'normalized_yxyx', 'normalized_cxcywh']. Will raise an
        error if not None and target_bboxes is None.
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        target_bboxes, target_class_ids = self._check_target_args(target_bboxes, target_bboxes_format, target_class_ids)

        for i, (prediction, target_bbox, target_class_id) in enumerate(zip(self._images_prediction_lst, target_bboxes, target_class_ids)):
            image_output_path = os.path.join(output_folder, f"pred_{i}.jpg")
            prediction.save(output_path=image_output_path, box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)


@dataclass
class VideoDetectionPrediction(VideoPredictions):
    """Object wrapping the list of image detection predictions as a Video.

    :attr _images_prediction_lst:   List of the predictions results
    :att fps:                       Frames per second of the video
    """

    _images_prediction_lst: List[ImageDetectionPrediction]
    fps: int

    def draw(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None) -> List[np.ndarray]:
        """Draw the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :return:                List of images with predicted bboxes. Note that this does not modify the original image.
        """
        frames_with_bbox = [
            result.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping) for result in self._images_prediction_lst
        ]
        return frames_with_bbox

    def show(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """Display the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        frames = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)
        show_video_from_frames(window_name="Detection", frames=frames, fps=self.fps)

    def save(self, output_path: str, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        frames = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)
        save_video(output_path=output_path, frames=frames, fps=self.fps)
