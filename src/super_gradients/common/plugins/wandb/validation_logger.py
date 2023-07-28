from typing import Optional

import torch
import numpy as np

from super_gradients.training.utils.callbacks import Callback, PhaseContext
from super_gradients.common.plugins.wandb.log_predictions import visualize_image_detection_prediction_on_wandb
from super_gradients.training.models.predictions import DetectionPrediction
from super_gradients.training.utils.predict import ImageDetectionPrediction
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.module_interfaces import HasPredict
from super_gradients.training.utils.utils import unwrap_model

try:
    import wandb
except (ModuleNotFoundError, ImportError, NameError):
    pass  # no action or logging - this is normal in most cases


class WandBDetectionValidationPredictionLoggerCallback(Callback):
    def __init__(
        self,
        class_names,
        max_predictions_plotted: Optional[int] = None,
        post_prediction_callback: Optional[DetectionPostPredictionCallback] = None,
        reverse_channels: bool = True,
    ) -> None:
        """A callback for logging object detection predictions to Weights & Biases during training. This callback is logging images on each batch in validation
        and accumulating generated images in a `wandb.Table` in the RAM. This could potentially cause OOM errors for very large datasets like COCO. In order to
        avoid this, it is recommended to explicitly set the parameter `max_predictions_plotted` to a small value, thus limiting the number of images logged in
        the table.

        :param class_names:                 A list of class names.
        :param max_predictions_plotted:     Maximum number of predictions to be plotted per epoch. This is set to `None` by default which means that the
                                            predictions corresponding to all images from `context.inputs` is logged, otherwise only `max_predictions_plotted`
                                            number of images is logged. Since `WandBDetectionValidationPredictionLoggerCallback` accumulates the generated
                                            images in the RAM, it is advisable that the value of this parameter be explicitly specified for larger datasets in
                                            order to avoid out-of-memory errors.
        :param post_prediction_callback:    `DetectionPostPredictionCallback` for post-processing outputs of the model.
        :param reverse_channels:            Reverse the order of channels on the images while plotting.
        """
        super().__init__()
        self.class_names = class_names
        self.max_predictions_plotted = max_predictions_plotted
        self.post_prediction_callback = post_prediction_callback
        self.reverse_channels = reverse_channels
        self.wandb_images = []
        self.epoch_count = 0
        self.mean_prediction_dicts = []
        self.wandb_table = wandb.Table(columns=["Epoch", "Prediction", "Mean-Confidence"])

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        self.wandb_images = []
        mean_prediction_dict = {class_name: 0.0 for class_name in self.class_names}
        if isinstance(context.net, HasPredict):
            post_nms_predictions = context.net(context.inputs)
        else:
            self.post_prediction_callback = (
                unwrap_model(context.net).get_post_prediction_callback() if self.post_prediction_callback is None else self.post_prediction_callback
            )
            self.post_prediction_callback.fuse_layers = False
            post_nms_predictions = self.post_prediction_callback(context.preds, device=context.device)
        if self.max_predictions_plotted is not None:
            post_nms_predictions = post_nms_predictions[: self.max_predictions_plotted]
            input_images = context.inputs[: self.max_predictions_plotted]
        else:
            input_images = context.inputs
        for prediction, image in zip(post_nms_predictions, input_images):
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
            wandb_image = visualize_image_detection_prediction_on_wandb(
                prediction=image_prediction, show_confidence=True, reverse_channels=self.reverse_channels
            )
            self.wandb_images.append(wandb_image)

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        for wandb_image, mean_prediction_dict in zip(self.wandb_images, self.mean_prediction_dicts):
            self.wandb_table.add_data(self.epoch_count, wandb_image, mean_prediction_dict)
        self.wandb_images, self.mean_prediction_dicts = [], []
        self.epoch_count += 1

    def on_training_end(self, context: PhaseContext) -> None:
        wandb.log({"Validation-Prediction": self.wandb_table})
