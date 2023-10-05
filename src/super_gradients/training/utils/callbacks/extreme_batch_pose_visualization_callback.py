from typing import Optional, Tuple, Callable, List, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics import Metric

from super_gradients.common.registry.registry import register_callback
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import undo_flat_collate_tensors_with_batch_index
from super_gradients.training.utils.callbacks import PhaseContext
from super_gradients.training.utils.callbacks.callbacks import ExtremeBatchCaseVisualizationCallback
from super_gradients.training.utils.distributed_training_utils import maybe_all_gather_np_images
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization


@register_callback("ExtremeBatchPoseEstimationVisualizationCallback")
class ExtremeBatchPoseEstimationVisualizationCallback(ExtremeBatchCaseVisualizationCallback):
    """
    ExtremeBatchPoseEstimationVisualizationCallback

    Visualizes worst/best batch in an epoch for Object detection.
    For clarity, the batch is saved twice in the SG Logger, once with the model's predictions and once with
     ground truth targets.

    Assumptions on bbox dormats:
     - After applying post_prediction_callback on context.preds, the predictions are a list/Tensor s.t:
        predictions[i] is a tensor of shape nx6 - (x1, y1, x2, y2, confidence, class) where x and y are in pixel units.

     - context.targets is a tensor of shape (total_num_targets, 6), in LABEL_CXCYWH format:  (index, label, cx, cy, w, h).



    Example usage in Yaml config:

        training_hyperparams:
          phase_callbacks:
            - ExtremeBatchDetectionVisualizationCallback:
                metric:
                  DetectionMetrics_050:
                    score_thres: 0.1
                    top_k_predictions: 300
                    num_cls: ${num_classes}
                    normalize_targets: True
                    post_prediction_callback:
                      _target_: super_gradients.training.models.detection_models.pp_yolo_e.PPYoloEPostPredictionCallback
                      score_threshold: 0.01
                      nms_top_k: 1000
                      max_predictions: 300
                      nms_threshold: 0.7
                metric_component_name: 'mAP@0.50'
                post_prediction_callback:
                  _target_: super_gradients.training.models.detection_models.pp_yolo_e.PPYoloEPostPredictionCallback
                  score_threshold: 0.25
                  nms_top_k: 1000
                  max_predictions: 300
                  nms_threshold: 0.7
                normalize_targets: True

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
        keypoint_colors: List[Tuple[int, int, int]],
        edge_colors: List[Tuple[int, int, int]],
        edge_links: List[Tuple[int, int]],
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
        self.keypoint_colors = OmegaConf.to_container(keypoint_colors)
        self.edge_colors = OmegaConf.to_container(edge_colors)
        self.edge_links = OmegaConf.to_container(edge_links)
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
    def visualize_batch(
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
    ):

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
        Processes the extreme batch, and returns 2 image batches for visualization - one with predictions and one with GT boxes.
        :return:Tuple[np.ndarray, np.ndarray], the predictions batch, the GT batch
        """
        inputs = self.universal_undo_preprocessing_fn(self.extreme_batch)
        target_boxes, target_joints, target_crowds = self.extreme_targets
        predictions = self.post_prediction_callback(self.extreme_preds, self.extreme_batch.device)

        images_to_save_preds = self.visualize_batch(
            inputs,
            keypoints=[p.poses for p in predictions],
            bboxes=[p.bboxes for p in predictions],
            scores=[p.scores for p in predictions],
            is_crowd=None,
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
            show_keypoint_confidence=True,
        )
        images_to_save_preds = np.stack(images_to_save_preds)

        batch_size = len(predictions)

        target_joints_unpacked = undo_flat_collate_tensors_with_batch_index(target_joints, batch_size)
        target_boxes_unpacked = undo_flat_collate_tensors_with_batch_index(target_boxes, batch_size)
        target_crowd_unpacked = undo_flat_collate_tensors_with_batch_index(target_crowds, batch_size)

        images_to_save_gt = self.visualize_batch(
            inputs,
            keypoints=target_joints_unpacked,
            bboxes=target_boxes_unpacked,
            scores=None,
            is_crowd=target_crowd_unpacked,
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
            show_keypoint_confidence=False,
        )
        images_to_save_gt = np.stack(images_to_save_gt)

        # Stack the predictions and GT images together
        return np.concatenate([images_to_save_gt, images_to_save_preds], axis=2)

    def on_train_loader_end(self, context: PhaseContext) -> None:
        if self.enable_on_train_loader and context.epoch % self.freq == 0:
            images_to_save_preds_with_gt = self.process_extreme_batch()
            images_to_save_preds_with_gt = maybe_all_gather_np_images(images_to_save_preds_with_gt)

            if self.max_images is not None:
                images_to_save_preds_with_gt = images_to_save_preds_with_gt[: self.max_images]

            if not context.ddp_silent_mode:
                context.sg_logger.add_images(tag=f"train/{self._tag}", images=images_to_save_preds_with_gt, global_step=context.epoch, data_format="NHWC")

            self._reset()

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        if self.enable_on_valid_loader and context.epoch % self.freq == 0:
            images_to_save_preds_with_gt = self.process_extreme_batch()
            images_to_save_preds_with_gt = maybe_all_gather_np_images(images_to_save_preds_with_gt)

            if self.max_images is not None:
                images_to_save_preds_with_gt = images_to_save_preds_with_gt[: self.max_images]

            if not context.ddp_silent_mode:
                context.sg_logger.add_images(tag=f"valid/{self._tag}", images=images_to_save_preds_with_gt, global_step=context.epoch, data_format="NHWC")

            self._reset()