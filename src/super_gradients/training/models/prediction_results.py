import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Iterator
from dataclasses import dataclass

import numpy as np

import wandb

from super_gradients.training.models.predictions import Prediction, DetectionPrediction
from super_gradients.training.utils.media.video import show_video_from_frames, save_video
from super_gradients.training.utils.media.image import show_image, save_image
from super_gradients.training.utils.visualization.utils import generate_color_mapping
from super_gradients.training.utils.visualization.detection import draw_bbox


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
class ImageDetectionPrediction(ImagePrediction):
    """Object wrapping an image and a detection model's prediction.

    :attr image:        Input image
    :attr predictions:  Predictions of the model
    :attr class_names:  List of the class names to predict
    """

    image: np.ndarray
    prediction: DetectionPrediction
    class_names: List[str]

    def draw(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """Draw the predicted bboxes on the image.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        :return:                Image with predicted bboxes. Note that this does not modify the original image.
        """
        image = self.image.copy()
        color_mapping = color_mapping or generate_color_mapping(len(self.class_names))

        for pred_i in range(len(self.prediction)):

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

        return image
    
    def visualize_on_wandb(self, show_confidence: bool = True):
        boxes = []
        image = self.image.copy()
        height, width, _ = image.shape
        class_id_to_labels = {
            int(_id): str(_class_name) for _id, _class_name in enumerate(self.class_names)
        }
        
        for pred_i in range(len(self.prediction)):
            class_id = int(self.prediction.labels[pred_i])
            box = {
                "position": {
                    "minX": float(int(self.prediction.bboxes_xyxy[pred_i, 0]) / width),
                    "maxX": float(int(self.prediction.bboxes_xyxy[pred_i, 2]) / width),
                    "minY": float(int(self.prediction.bboxes_xyxy[pred_i, 1]) / height),
                    "maxY": float(int(self.prediction.bboxes_xyxy[pred_i, 3]) / height),
                },
                "class_id": int(class_id),
                "box_caption": str(self.class_names[class_id])
            }
            if show_confidence:
                box["scores"] = {
                    "confidence": float(round(self.prediction.confidence[pred_i], 2))
                }
            boxes.append(box)
        
        wandb_image = wandb.Image(image, boxes={
            "predictions": {
                "box_data": boxes,
                "class_labels": class_id_to_labels
            }
        })
        
        return wandb_image, class_id_to_labels

    def show(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None, visualize_on_wandb: bool = False) -> None:
        """Display the image with predicted bboxes.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        image = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)
        show_image(image)
        
        if visualize_on_wandb:
            if wandb.run is None:
                raise wandb.Error("You must call `wandb.init()` before calling `ImageDetectionPrediction.show(visualize_on_wandb=True)`")
            wandb_image, class_id_to_labels = self.visualize_on_wandb(show_confidence=show_confidence)
            wandb.log({"Predictions": wandb_image})

    def save(self, output_path: str, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """Save the predicted bboxes on the images.

        :param output_path:     Path to the output video file.
        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        image = self.draw(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping)
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
class ImagesDetectionPrediction(ImagesPredictions):
    """Object wrapping the list of image detection predictions.

    :attr _images_prediction_lst:  List of the predictions results
    """

    _images_prediction_lst: List[ImageDetectionPrediction]

    def show(self, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None, visualize_on_wandb: bool = False) -> None:
        """Display the predicted bboxes on the images.

        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        for prediction in self._images_prediction_lst:
            prediction.show(box_thickness=box_thickness, show_confidence=show_confidence, color_mapping=color_mapping, visualize_on_wandb=visualize_on_wandb)

    def save(
        self, output_folder: str, box_thickness: int = 2, show_confidence: bool = True, color_mapping: Optional[List[Tuple[int, int, int]]] = None
    ) -> None:
        """Save the predicted bboxes on the images.

        :param output_folder:     Folder path, where the images will be saved.
        :param box_thickness:   Thickness of bounding boxes.
        :param show_confidence: Whether to show confidence scores on the image.
        :param color_mapping:   List of tuples representing the colors for each class.
                                Default is None, which generates a default color mapping based on the number of class names.
        """
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        for i, prediction in enumerate(self._images_prediction_lst):
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
