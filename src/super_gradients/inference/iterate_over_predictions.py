from typing import Iterable, Tuple, Union

import numpy as np
import torch

__all__ = ["iterate_over_detection_predictions_in_batched_format", "iterate_over_detection_predictions_in_flat_format"]


NumpyArrayOrTensor = Union[np.ndarray, torch.Tensor]


def iterate_over_detection_predictions_in_flat_format(predictions: NumpyArrayOrTensor, batch_size: int):
    """
    Iterate over object detection predictions in flat format.
    This method is suitable for iterating over predictions of object detection models exported to ONNX format
    with postprocessing. An exported object detection model can have 'flat' or 'batched' output format.
    A flat output format means that all detections from all images in batch are concatenated together and
    image index is added as a first column.

    >>> predictions = model(batch_of_images)
    >>> for image_detections in iterate_over_detection_predictions_in_flat_format(predictions, len(batch_of_images)):
    >>>     image_index, pred_bboxes, pred_scores, pred_labels = image_detections
    >>>     # Do something with predictions for image with index image_index
    >>>     ...

    :param predictions: An array of [N, 7] shape where N is a total number of detections in batch.
                        Each detection is represented by [image_index, x1, y1, x2, y2, score, label] values.
    :param batch_size:  A number of images in batch. This must be passed explicitly because batch size
                        cannot be inferred from predictions array.
    :return:            A generator that yields (image_index, bboxes, scores, labels) for each image in batch
                        image_index: An index of image in batch
                        bboxes: A 2D array of shape (num_predictions, 4) containing bounding boxes in format (x1, y1, x2, y2)
                        scores: A 1D array of shape (num_predictions,) containing class scores
                        labels: A 1D array of shape (num_predictions,) containing class labels. Class labels casted to int.
    """
    if isinstance(predictions, (list, tuple)) and len(predictions) == 1:
        [predictions] = predictions

    if not isinstance(predictions, (np.ndarray, torch.Tensor)):
        raise ValueError(f"predictions must be a tensor or numpy array, got {type(predictions)}")

    if len(predictions.shape) != 2 or predictions.shape[1] != 7:
        raise ValueError(f"predictions must be a tensor or numpy array of shape (num_predictions, 7), got {predictions.shape}")

    for image_index in range(batch_size):
        mask = predictions[:, 0] == image_index
        pred_bboxes = predictions[mask, 1:5]
        pred_scores = predictions[mask, 5]
        pred_labels = predictions[mask, 6]

        if torch.is_tensor(pred_labels):
            pred_labels = pred_labels.long()
        else:
            pred_labels = pred_labels.astype(int)

        yield image_index, pred_bboxes, pred_scores, pred_labels


def iterate_over_detection_predictions_in_batched_format(
    predictions: Tuple[NumpyArrayOrTensor, NumpyArrayOrTensor, NumpyArrayOrTensor, NumpyArrayOrTensor]
) -> Iterable[Tuple[int, NumpyArrayOrTensor, NumpyArrayOrTensor, NumpyArrayOrTensor]]:
    """
    Iterate over object detection predictions in batched format.
    This method is suitable for iterating over predictions of object detection models exported to ONNX format
    with postprocessing. An exported object detection model can have 'flat' or 'batched' output format.
    A batched output format means that all detections from all images in batch are padded and stacked together.
    So one should iterate over all detections and filter out detections for each image separately which this method does.

    >>> predictions = model(batch_of_images)
    >>> for image_detections in iterate_over_detection_predictions_in_batched_format(predictions):
    >>>     image_index, pred_bboxes, pred_scores, pred_labels = image_detections
    >>>     # Do something with predictions for image with index image_index
    >>>     ...

    :param predictions:    A tuple of (num_detections, bboxes, scores, labels)
           num_detections: A 1D array of shape (batch_size,) containing number of detections per image
           bboxes:         A 3D array of shape (batch_size, max_detections, 4) containing bounding boxes in format (x1, y1, x2, y2)
           scores:         A 2D array of shape (batch_size, max_detections) containing class scores
           labels:         A 2D array of shape (batch_size, max_detections) containing class labels
    :return:               A generator that yields (image_index, bboxes, scores, labels) for each image in batch
                           image_index: An index of image in batch
                           bboxes: A 2D array of shape (num_predictions, 4) containing bounding boxes in format (x1, y1, x2, y2)
                           scores: A 1D array of shape (num_predictions,) containing class scores
                           labels: A 1D array of shape (num_predictions,) containing class labels. Class labels casted to int.

    """
    num_detections, detected_bboxes, detected_scores, detected_labels = predictions
    num_detections = num_detections.reshape(-1)
    batch_size = len(num_detections)

    detected_bboxes = detected_bboxes.reshape(batch_size, -1, 4)
    detected_scores = detected_scores.reshape(batch_size, -1)
    detected_labels = detected_labels.reshape(batch_size, -1)

    if torch.is_tensor(detected_labels):
        detected_labels = detected_labels.long()
    else:
        detected_labels = detected_labels.astype(int)

    for image_index in range(batch_size):
        num_detection_in_image = num_detections[image_index]

        pred_bboxes = detected_bboxes[image_index, :num_detection_in_image]
        pred_scores = detected_scores[image_index, :num_detection_in_image]
        pred_labels = detected_labels[image_index, :num_detection_in_image]

        yield image_index, pred_bboxes, pred_scores, pred_labels
