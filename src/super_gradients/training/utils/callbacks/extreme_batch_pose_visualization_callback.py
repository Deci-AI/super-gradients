import typing
from typing import Optional, Tuple, Callable, List, Union

import numpy as np
import torch
from omegaconf import ListConfig
from torch import Tensor
from torchmetrics import Metric

from super_gradients.common.registry.registry import register_callback
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy
from super_gradients.training.utils.callbacks.callbacks import ExtremeBatchCaseVisualizationCallback
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization

# These imports are required for type hints and not used anywhere else
# Wrapping them under typing.TYPE_CHECKING is a legit way to avoid circular imports
# while still having type hints
if typing.TYPE_CHECKING:
    from super_gradients.training.samples import PoseEstimationSample
    from super_gradients.module_interfaces import PoseEstimationPredictions


@register_callback("ExtremeBatchPoseEstimationVisualizationCallback")
class ExtremeBatchPoseEstimationVisualizationCallback(ExtremeBatchCaseVisualizationCallback):
    """
    ExtremeBatchPoseEstimationVisualizationCallback

    Visualizes worst/best batch in an epoch for pose estimation task.
    This class visualize horizontally-stacked GT and predicted poses.
    It requires a key 'gt_samples' (List[PoseEstimationSample]) to be present in additional_batch_items dictionary.

    Supported models: YoloNASPose
    Supported datasets: COCOPoseEstimationDataset

    Example usage in Yaml config:

        training_hyperparams:
          phase_callbacks:
              - ExtremeBatchPoseEstimationVisualizationCallback:
                  keypoint_colors: ${dataset_params.keypoint_colors}
                  edge_colors: ${dataset_params.edge_colors}
                  edge_links: ${dataset_params.edge_links}
                  loss_to_monitor: YoloNASPoseLoss/loss
                  max: True
                  freq: 1
                  max_images: 16
                  enable_on_train_loader: True
                  enable_on_valid_loader: True
                  post_prediction_callback:
                    _target_: super_gradients.training.models.pose_estimation_models.yolo_nas_pose.YoloNASPosePostPredictionCallback
                    pose_confidence_threshold: 0.01
                    nms_iou_threshold: 0.7
                    pre_nms_max_predictions: 300
                    post_nms_max_predictions: 30

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

    :param classes: List[str], a list of class names corresponding to the class indices for display.
     When None, will try to fetch this through a "classes" attribute of the valdiation dataset. If such attribute does
      not exist an error will be raised (default=None).

    :param normalize_targets: bool, whether to scale the target bboxes. If the bboxes returned by the validation data loader
     are in pixel values range, this needs to be set to True (default=False)

    """

    def __init__(
        self,
        post_prediction_callback: Callable,
        keypoint_colors: Union[ListConfig, List[Tuple[int, int, int]]],
        edge_colors: Union[ListConfig, List[Tuple[int, int, int]]],
        edge_links: Union[ListConfig, List[Tuple[int, int]]],
        metric: Optional[Metric] = None,
        metric_component_name: Optional[str] = None,
        loss_to_monitor: Optional[str] = None,
        max: bool = False,
        freq: int = 1,
        max_images: Optional[int] = None,
        enable_on_train_loader: bool = False,
        enable_on_valid_loader: bool = True,
    ):
        super().__init__(
            metric=metric,
            metric_component_name=metric_component_name,
            loss_to_monitor=loss_to_monitor,
            max=max,
            freq=freq,
            enable_on_train_loader=enable_on_train_loader,
            enable_on_valid_loader=enable_on_valid_loader,
        )
        self.post_prediction_callback = post_prediction_callback
        self.keypoint_colors = list(tuple(map(int, color)) for color in keypoint_colors)
        self.edge_colors = list(tuple(map(int, color)) for color in edge_colors)
        self.edge_links = list(tuple(map(int, link)) for link in edge_links)
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
        keypoints: List[Union[np.ndarray, Tensor]],
        bboxes: List[Union[None, np.ndarray, Tensor]],
        scores: Optional[List[Union[None, np.ndarray, Tensor]]],
        is_crowd: Optional[List[Union[None, np.ndarray, Tensor]]],
        keypoint_colors: List[Tuple[int, int, int]],
        edge_colors: List[Tuple[int, int, int]],
        edge_links: List[Tuple[int, int]],
        show_keypoint_confidence: bool,
    ) -> List[np.ndarray]:
        """
        Generate list of samples visualization of a batch of images with keypoints and bounding boxes.

        :param image_tensor:             Images batch of [Batch Size, 3, H, W] shape with values in [0, 255] range.
                                         The images should be scaled to [0, 255] range and converted to uint8 type beforehead.
        :param keypoints:                Keypoints in XY format. Shape [Num Instances, Num Joints, 2]. Can be None.
        :param bboxes:                   Bounding boxes in XYXY format. Shape [Num Instances, 4]. Can be None.
        :param scores:                   Keypoint scores. Shape [Num Instances, Num Joints]. Can be None.
        :param is_crowd:                 Whether each sample is crowd or not. Shape [Num Instances]. Can be None.
        :param keypoint_colors:          Keypoint colors. Shape [Num Joints, 3]
        :param edge_colors:              Edge colors between joints. Shape [Num Links, 3]
        :param edge_links:               Edge links between joints. Shape [Num Links, 2]
        :param show_keypoint_confidence: Whether to show confidence for each keypoint. Requires `scores` to be not None.
        :return:                         List of visualization images.
        """

        out_images = []
        for i in range(image_tensor.shape[0]):
            keypoints_i = keypoints[i]
            bboxes_i = bboxes[i]
            scores_i = scores[i] if scores is not None else None
            is_crowd_i = is_crowd[i] if is_crowd is not None else None

            if torch.is_tensor(keypoints_i):
                keypoints_i = keypoints_i.detach().cpu().numpy()
            if torch.is_tensor(bboxes_i):
                bboxes_i = bboxes_i.detach().cpu().numpy()
            if torch.is_tensor(scores_i):
                scores_i = scores_i.detach().cpu().numpy()
            if torch.is_tensor(is_crowd_i):
                is_crowd_i = is_crowd_i.detach().cpu().numpy()

            res_image = image_tensor[i]
            res_image = PoseVisualization.draw_poses(
                image=res_image,
                poses=keypoints_i,
                boxes=bboxes_i,
                scores=scores_i,
                is_crowd=is_crowd_i,
                show_keypoint_confidence=show_keypoint_confidence,
                edge_links=edge_links,
                edge_colors=edge_colors,
                keypoint_colors=keypoint_colors,
                keypoint_confidence_threshold=0.01,
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
        gt_samples: List[PoseEstimationSample] = self.extreme_additional_batch_items["gt_samples"]
        predictions: List[PoseEstimationPredictions] = self.post_prediction_callback(self.extreme_preds)

        images_to_save_preds = self._visualize_batch(
            image_tensor=inputs,
            keypoints=[p.poses for p in predictions],
            bboxes=[p.bboxes_xyxy for p in predictions],
            scores=[p.scores for p in predictions],
            is_crowd=None,
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
            show_keypoint_confidence=True,
        )
        images_to_save_preds = np.stack(images_to_save_preds)

        images_to_save_gt = self._visualize_batch(
            image_tensor=inputs,
            keypoints=[gt.joints for gt in gt_samples],
            bboxes=[xywh_to_xyxy(gt.bboxes_xywh, image_shape=None) if gt.bboxes_xywh is not None else None for gt in gt_samples],
            scores=None,
            is_crowd=[gt.is_crowd for gt in gt_samples],
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
            show_keypoint_confidence=False,
        )
        images_to_save_gt = np.stack(images_to_save_gt)

        # Stack the predictions and GT images together
        return np.concatenate([images_to_save_gt, images_to_save_preds], axis=2)
