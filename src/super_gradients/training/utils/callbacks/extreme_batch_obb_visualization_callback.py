import typing
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from super_gradients.common.registry.registry import register_callback
from super_gradients.training.utils.callbacks.callbacks import ExtremeBatchCaseVisualizationCallback
from super_gradients.training.utils.visualization.obb import OBBVisualization
from super_gradients.training.utils.visualization.utils import generate_color_mapping

# These imports are required for type hints and not used anywhere else
# Wrapping them under typing.TYPE_CHECKING is a legit way to avoid circular imports
# while still having type hints
if typing.TYPE_CHECKING:
    from super_gradients.training.datasets.obb.dota import OBBSample
    from super_gradients.module_interfaces.obb_predictions import AbstractOBBPostPredictionCallback, OBBPredictions


@register_callback()
class ExtremeBatchOBBVisualizationCallback(ExtremeBatchCaseVisualizationCallback):
    """
    ExtremeBatchOBBVisualizationCallback

    Visualizes worst/best batch in an epoch for OBB detection task.
    This class visualize horizontally-stacked GT and predicted boxes.
    It requires a key 'gt_samples' (List[OBBSample]) to be present in additional_batch_items dictionary.

    Supported models: YoloNAS-R
    Supported datasets: DOTAOBBDataset

    Example usage in Yaml config:

        training_hyperparams:
          phase_callbacks:
              - ExtremeBatchOBBVisualizationCallback:
                  loss_to_monitor: YoloNASRLoss/loss
                  max: True
                  freq: 1
                  max_images: 16
                  enable_on_train_loader: True
                  enable_on_valid_loader: True
                  post_prediction_callback:
                    _target_: super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_post_prediction_callback.YoloNASRPostPredictionCallback
                    score_threshold: 0.25
                    pre_nms_max_predictions: 4096
                    post_nms_max_predictions: 512
                    nms_iou_threshold: 0.6

    :param metric: Metric, will be the metric which is monitored.

    :param metric_component_name: In case metric returns multiple values (as Mapping),
     the value at metric.compute()[metric_component_name] will be the one monitored.

    :param loss_to_monitor: str, loss_to_monitor corresponding to the 'criterion' passed through training_params in Trainer.train(...).
     Monitoring loss follows the same logic as metric_to_watch in Trainer.train(..), when watching the loss and should be:

        if hasattr(criterion, "component_names") and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"<COMPONENT_NAME>.

        If a single item is returned rather then a tuple:
            <LOSS_CLASS.__name__>.

        When there is no such attributes and criterion.forward(..) returns a tuple:
            <LOSS_CLASS.__name__>"/"Loss_"<IDX>

    :param max: bool, Whether to take the batch corresponding to the max value of the metric/loss or
     the minimum (default=False).

    :param freq: int, epoch frequency to perform all of the above (default=1).
    """

    def __init__(
        self,
        post_prediction_callback: "AbstractOBBPostPredictionCallback",
        class_names: List[str],
        class_colors=None,
        metric: Optional[Metric] = None,
        metric_component_name: Optional[str] = None,
        loss_to_monitor: Optional[str] = None,
        max: bool = False,
        freq: int = 1,
        max_images: int = -1,
        enable_on_train_loader: bool = False,
        enable_on_valid_loader: bool = True,
    ):
        if class_colors is None:
            class_colors = generate_color_mapping(num_classes=len(class_names))

        super().__init__(
            metric=metric,
            metric_component_name=metric_component_name,
            loss_to_monitor=loss_to_monitor,
            max=max,
            freq=freq,
            enable_on_train_loader=enable_on_train_loader,
            enable_on_valid_loader=enable_on_valid_loader,
        )
        self.class_names = list(class_names)
        self.class_colors = class_colors
        self.post_prediction_callback = post_prediction_callback
        self.max_images = max_images

    @classmethod
    def universal_undo_preprocessing_fn(cls, inputs: torch.Tensor) -> np.ndarray:
        """
        A universal reversing of preprocessing to be passed to DetectionVisualization.visualize_batch's undo_preprocessing_func kwarg.
        :param inputs:
        :return:
        """
        inputs = inputs - inputs.min()
        inputs /= inputs.max()
        inputs *= 255
        inputs = inputs.to(torch.uint8)
        inputs = inputs.cpu().numpy()
        inputs = inputs[:, ::-1, :, :].transpose(0, 2, 3, 1)
        inputs = np.ascontiguousarray(inputs, dtype=np.uint8)
        return inputs

    @classmethod
    def _visualize_batch(
        cls,
        image_tensor: np.ndarray,
        rboxes: List[Union[np.ndarray, Tensor]],
        labels: List[Union[np.ndarray, Tensor]],
        scores: Optional[List[Union[np.ndarray, Tensor]]],
        class_colors: List[Tuple[int, int, int]],
        class_names: List[str],
    ) -> List[np.ndarray]:
        """
        Generate list of samples visualization of a batch of images with keypoints and bounding boxes.

        :param image_tensor:             Images batch of [Batch Size, 3, H, W] shape with values in [0, 255] range.
                                         The images should be scaled to [0, 255] range and converted to uint8 type beforehead.
        :param scores:                   Keypoint scores. Shape [Num Instances, Num Joints]. Can be None.
        :return:                         List of visualization images.
        """

        out_images = []
        for i in range(image_tensor.shape[0]):
            rboxes_i = rboxes[i]
            labels_i = labels[i]
            scores_i = scores[i] if scores is not None else None

            if torch.is_tensor(rboxes_i):
                rboxes_i = rboxes_i.detach().cpu().numpy()
            if torch.is_tensor(labels_i):
                labels_i = labels_i.detach().cpu().numpy()
            if torch.is_tensor(scores_i):
                scores_i = scores_i.detach().cpu().numpy()

            res_image = image_tensor[i]
            res_image = OBBVisualization.draw_obb(
                image=res_image,
                rboxes_cxcywhr=rboxes_i,
                labels=labels_i,
                scores=scores_i,
                class_colors=class_colors,
                class_names=class_names,
                show_confidence=True,
                show_labels=True,
            )

            out_images.append(res_image)

        return out_images

    @torch.no_grad()
    def process_extreme_batch(self) -> np.ndarray:
        """
        Processes the extreme batch, and returns batche of images for visualization - predictions and GT poses stacked horizontally.

        :return: np.ndarray - the visualization of predictions and GT
        """
        if "gt_samples" not in self.extreme_additional_batch_items:
            raise RuntimeError(
                "ExtremeBatchPoseEstimationVisualizationCallback requires 'gt_samples' to be present in additional_batch_items."
                "Currently only YoloNASPose model is supported. Old DEKR recipe is not supported at the moment."
            )

        inputs = self.universal_undo_preprocessing_fn(self.extreme_batch)
        gt_samples: List["OBBSample"] = self.extreme_additional_batch_items["gt_samples"]
        predictions: List["OBBPredictions"] = self.post_prediction_callback(self.extreme_preds)

        images_to_save_preds = self._visualize_batch(
            image_tensor=inputs,
            rboxes=[p.rboxes_cxcywhr for p in predictions],
            labels=[p.labels for p in predictions],
            scores=[p.scores for p in predictions],
            class_colors=self.class_colors,
            class_names=self.class_names,
        )
        images_to_save_preds = np.stack(images_to_save_preds)

        images_to_save_gt = self._visualize_batch(
            image_tensor=inputs,
            rboxes=[gt.rboxes_cxcywhr for gt in gt_samples],
            labels=[gt.labels for gt in gt_samples],
            scores=None,
            class_colors=self.class_colors,
            class_names=self.class_names,
        )
        images_to_save_gt = np.stack(images_to_save_gt)

        # Stack the predictions and GT images together
        return np.concatenate([images_to_save_gt, images_to_save_preds], axis=2)
