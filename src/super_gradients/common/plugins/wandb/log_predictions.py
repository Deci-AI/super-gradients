try:
    import wandb
except (ModuleNotFoundError, ImportError, NameError):
    pass  # no action or logging - this is normal in most cases

import numpy as np
from tqdm import tqdm

from super_gradients.training.transforms.transforms import DetectionTargetsFormatTransform
from super_gradients.training.datasets.data_formats.default_formats import XYXY_LABEL
from super_gradients.training.datasets.detection_datasets import DetectionDataset

from super_gradients.training.utils.predict import ImageDetectionPrediction, ImagesDetectionPrediction


def visualize_image_detection_prediction_on_wandb(prediction: ImageDetectionPrediction, show_confidence: bool, reverse_channels: bool = False):
    """Visualize detection results on a single image.

    :param prediction:          Prediction results of a single image
                                (a `super_gradients.training.models.prediction_results.ImageDetectionPrediction` object)
    :param show_confidence:     Whether to log confidence scores to Weights & Biases or not.
    :param reverse_channels:    Reverse the order of channels on the images while plotting.
    """
    boxes = []
    image = prediction.image.copy()
    image = image[:, :, ::-1] if reverse_channels else image
    height, width, _ = image.shape
    class_id_to_labels = {int(_id): str(_class_name) for _id, _class_name in enumerate(prediction.class_names)}

    for pred_i in range(len(prediction.prediction)):
        class_id = int(prediction.prediction.labels[pred_i])
        box = {
            "position": {
                "minX": float(int(prediction.prediction.bboxes_xyxy[pred_i, 0]) / width),
                "maxX": float(int(prediction.prediction.bboxes_xyxy[pred_i, 2]) / width),
                "minY": float(int(prediction.prediction.bboxes_xyxy[pred_i, 1]) / height),
                "maxY": float(int(prediction.prediction.bboxes_xyxy[pred_i, 3]) / height),
            },
            "class_id": int(class_id),
            "box_caption": str(prediction.class_names[class_id]),
        }
        if show_confidence:
            box["scores"] = {"confidence": float(round(prediction.prediction.confidence[pred_i], 2))}
        boxes.append(box)

    return wandb.Image(image, boxes={"predictions": {"box_data": boxes, "class_labels": class_id_to_labels}})


def log_detection_results_to_wandb(prediction: ImagesDetectionPrediction, show_confidence: bool = True):
    """Log predictions for object detection to Weights & Biases using interactive bounding box overlays.

    :param prediction:        The model predictions (a `super_gradients.training.models.prediction_results.ImagesDetectionPrediction` object)
    :param show_confidence:   Whether to log confidence scores to Weights & Biases or not.
    """
    if wandb.run is None:
        raise wandb.Error("Images and bounding boxes cannot be visualized on Weights & Biases without initializing a run using `wandb.init()`")
    for prediction in prediction._images_prediction_lst:
        wandb_image = visualize_image_detection_prediction_on_wandb(prediction=prediction, show_confidence=show_confidence)
        wandb.log({"Predictions": wandb_image})


def plot_detection_dataset_on_wandb(detection_dataset: DetectionDataset, max_examples: int = None, dataset_name: str = None, reverse_channels: bool = True):
    """Log a detection dataset to Weights & Biases Table.

    :param detection_dataset:       The Detection Dataset (a `super_gradients.training.datasets.detection_datasets.DetectionDataset` object)
    :param max_examples:            Maximum number of examples from the detection dataset to plot (an `int`).
    :param dataset_name:            Name of the dataset (a `str`).
    :param reverse_channels:        Reverse the order of channels on the images while plotting.
    """
    max_examples = len(detection_dataset) if max_examples is None else max_examples
    wandb_table = wandb.Table(columns=["Images", "Class-Frequencies"])
    input_format = detection_dataset.output_target_format
    target_format_transform = DetectionTargetsFormatTransform(input_format=input_format, output_format=XYXY_LABEL)
    class_id_to_labels = {int(_id): str(_class_name) for _id, _class_name in enumerate(detection_dataset.classes)}
    for data_idx in tqdm(range(max_examples), desc="Plotting Examples on Weights & Biases"):
        image, targets, *_ = detection_dataset[data_idx]
        image = image.transpose(1, 2, 0).astype(np.int32)
        sample = target_format_transform({"image": image, "target": targets})
        boxes = sample["target"][:, 0:4]
        boxes = boxes[(boxes != 0).any(axis=1)]
        classes = targets[:, 0].tolist()
        wandb_boxes = []
        class_frequencies = {str(_class_name): 0 for _id, _class_name in enumerate(detection_dataset.classes)}
        for idx in range(boxes.shape[0]):
            wandb_boxes.append(
                {
                    "position": {
                        "minX": float(boxes[idx][0] / image.shape[1]),
                        "maxX": float(boxes[idx][2] / image.shape[1]),
                        "minY": float(boxes[idx][1] / image.shape[0]),
                        "maxY": float(boxes[idx][3] / image.shape[0]),
                    },
                    "class_id": int(classes[idx]),
                    "box_caption": str(class_id_to_labels[int(classes[idx])]),
                }
            )
            class_frequencies[str(class_id_to_labels[int(classes[idx])])] += 1
        image = image[:, :, ::-1] if reverse_channels else image
        wandb_table.add_data(wandb.Image(image, boxes={"ground_truth": {"box_data": wandb_boxes, "class_labels": class_id_to_labels}}), class_frequencies)
    dataset_name = "Dataset" if dataset_name is None else dataset_name
    wandb.log({dataset_name: wandb_table}, commit=False)
