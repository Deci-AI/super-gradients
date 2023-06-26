try:
    import wandb
except (ModuleNotFoundError, ImportError, NameError):
    pass  # no action or logging - this is normal in most cases

from super_gradients.training.utils.predict import ImageDetectionPrediction, ImagesDetectionPrediction


def _visualize_image_detection_prediction_on_wandb(prediction: ImageDetectionPrediction, show_confidence: bool):
    boxes = []
    image = prediction.image.copy()
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

    wandb_image = wandb.Image(image, boxes={"predictions": {"box_data": boxes, "class_labels": class_id_to_labels}})

    wandb.log({"Predictions": wandb_image})


def log_detection_results_to_wandb(prediction: ImagesDetectionPrediction, show_confidence: bool = True):
    """Log predictions for object detection to Weights & Biases using interactive bounding box overlays.

    :param prediction:        The model predictions (a `super_gradients.training.models.prediction_results.ImagesDetectionPrediction` object)
    :param show_confidence:   Whether to log confidence scores to Weights & Biases or not.
    """
    if wandb.run is None:
        raise wandb.Error("Images and bounding boxes cannot be visualized on Weights & Biases without initializing a run using `wandb.init()`")
    for prediction in prediction._images_prediction_lst:
        _visualize_image_detection_prediction_on_wandb(prediction=prediction, show_confidence=show_confidence)
