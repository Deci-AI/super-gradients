from typing import List, Tuple

import torch
import torchvision
from torch import Tensor

from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback, PoseEstimationPredictions


class YoloNASPosePostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """
    A post-prediction callback for YoloNASPose model.
    Performs confidence thresholding, Top-K and NMS steps.
    """

    def __init__(
        self,
        pose_confidence_threshold: float,
        nms_iou_threshold: float,
        pre_nms_max_predictions: int,
        post_nms_max_predictions: int,
    ):
        """
        :param pose_confidence_threshold: Pose detection confidence threshold
        :param nms_iou_threshold:         IoU threshold for NMS step.
        :param pre_nms_max_predictions:   Number of predictions participating in NMS step
        :param post_nms_max_predictions:  Maximum number of boxes to return after NMS step
        """
        if post_nms_max_predictions > pre_nms_max_predictions:
            raise ValueError("post_nms_max_predictions must be less than pre_nms_max_predictions")

        super().__init__()
        self.pose_confidence_threshold = pose_confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions

    @torch.no_grad()
    def __call__(self, outputs: Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], ...]) -> List[PoseEstimationPredictions]:
        """
        Take YoloNASPose's predictions and decode them into usable pose predictions.

        :param outputs: Output of the model's forward() method
        :return:        List of decoded predictions for each image in the batch.
        """
        # First is model predictions, second element of tuple is logits for loss computation
        predictions = outputs[0]

        decoded_predictions: List[PoseEstimationPredictions] = []
        for pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores in zip(*predictions):
            # pred_bboxes [Anchors, 4] in XYXY format
            # pred_scores [Anchors, 1] confidence scores [0..1]
            # pred_pose_coords [Anchors, Num Keypoints, 2] in (x,y) format
            # pred_pose_scores [Anchors, Num Keypoints] confidence scores [0..1]

            pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
            conf_mask = pred_bboxes_conf >= self.pose_confidence_threshold  # [Anchors]

            pred_bboxes_conf = pred_bboxes_conf[conf_mask].float()
            pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask].float()
            pred_pose_coords = pred_pose_coords[conf_mask].float()
            pred_pose_scores = pred_pose_scores[conf_mask].float()

            # Filter all predictions by self.nms_top_k
            if pred_bboxes_conf.size(0) > self.pre_nms_max_predictions:
                topk_candidates = torch.topk(pred_bboxes_conf, k=self.pre_nms_max_predictions, largest=True, sorted=True)
                pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
                pred_bboxes_xyxy = pred_bboxes_xyxy[topk_candidates.indices]
                pred_pose_coords = pred_pose_coords[topk_candidates.indices]
                pred_pose_scores = pred_pose_scores[topk_candidates.indices]

            # NMS
            idx_to_keep = torchvision.ops.boxes.nms(boxes=pred_bboxes_xyxy, scores=pred_bboxes_conf, iou_threshold=self.nms_iou_threshold)

            final_bboxes = pred_bboxes_xyxy[idx_to_keep]  # [Instances,]
            final_scores = pred_bboxes_conf[idx_to_keep]  # [Instances,]

            final_poses = torch.cat(
                [
                    pred_pose_coords[idx_to_keep],
                    pred_pose_scores[idx_to_keep].unsqueeze(-1),
                ],
                dim=-1,
            )  # [Instances, Num Keypoints, 3]

            decoded_predictions.append(
                PoseEstimationPredictions(
                    poses=final_poses[: self.post_nms_max_predictions],
                    scores=final_scores[: self.post_nms_max_predictions],
                    bboxes_xyxy=final_bboxes[: self.post_nms_max_predictions],
                )
            )

        return decoded_predictions
