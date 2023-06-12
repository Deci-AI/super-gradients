from super_gradients.training.utils.callbacks.base_callbacks import PhaseContext
import wandb
import torch
import numpy as np

from super_gradients.training.utils.callbacks import Callback, PhaseContext
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.common.plugins.wandb.log_predictions import visualize_image_detection_prediction_on_wandb
from super_gradients.training.models.predictions import DetectionPrediction
from super_gradients.training.models.prediction_results import ImageDetectionPrediction


class WandBDetectionValidationPredictionLoggerCallback(Callback):
    def __init__(
        self,
        class_names,
        score_threshold: float = 0.001,
        nms_threshold: float = 0.6,
        nms_top_k: int = 1000,
        max_predictions: int = 300,
        multi_label_per_box: bool = True,
    ) -> None:
        """A callback for logging object detection predictions to Weights & Biases during training.

        :param class_names:         A list of class names.
        :param score_threshold:     Predictions confidence threshold. Predictions with score lower than score_threshold will not participate in Top-K & NMS
        :param iou:                 IoU threshold for NMS step.
        :param nms_top_k:           Number of predictions participating in NMS step
        :param max_predictions:     maximum number of boxes to return after NMS step
        """
        super().__init__()
        self.class_names = class_names
        self.post_prediction_callback = PPYoloEPostPredictionCallback(
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            nms_top_k=nms_top_k,
            max_predictions=max_predictions,
            multi_label_per_box=multi_label_per_box,
        )
        self.wandb_images = []
        self.epoch_count = 0
        self.mean_prediction_dicts = []
        self.wandb_table = wandb.Table(columns=["Epoch", "Prediction", "Mean-Confidence"])

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        self.wandb_images = []
        mean_prediction_dict = {class_name: 0.0 for class_name in self.class_names}
        post_nms_predictions = self.post_prediction_callback(context.preds, device=context.device)
        for prediction, image in zip(post_nms_predictions, context.inputs):
            prediction = prediction if prediction is not None else torch.zeros((0, 6), dtype=torch.float32)
            prediction = prediction.detach().cpu().numpy()
            postprocessed_image = image.detach().cpu().numpy().transpose(1, 2, 0).astype(np.int32)
            image_prediction = ImageDetectionPrediction(
                image=postprocessed_image,
                class_names=self.class_names,
                prediction=DetectionPrediction(
                    bboxes=prediction[:, :4],
                    confidence=prediction[:, 4],
                    labels=prediction[:, 5],
                    bbox_format="xyxy",
                    image_shape=image.shape,
                ),
            )
            for predicted_label, prediction_confidence in zip(prediction[:, 5], prediction[:, 4]):
                mean_prediction_dict[self.class_names[int(predicted_label)]] += prediction_confidence
            mean_prediction_dict = {k: v / len(prediction[:, 4]) for k, v in mean_prediction_dict.items()}
            self.mean_prediction_dicts.append(mean_prediction_dict)
            wandb_image = visualize_image_detection_prediction_on_wandb(prediction=image_prediction, show_confidence=True)
            self.wandb_images.append(wandb_image)

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        for wandb_image, mean_prediction_dict in zip(self.wandb_images, self.mean_prediction_dicts):
            self.wandb_table.add_data(self.epoch_count, wandb_image, mean_prediction_dict)
        self.wandb_images, self.mean_prediction_dicts = [], []
        self.epoch_count += 1
    
    def on_training_end(self, context: PhaseContext) -> None:
        wandb.log({"Validation-Prediction": self.wandb_table})
