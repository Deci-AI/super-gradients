import torch
import torchvision


class YoloNASPosePostPredictionCallback:
    """
    A post-prediction callback for YoloNASPose model.
    Performs confidence thresholding, Top-K and NMS steps.
    """

    def __init__(self, score_threshold: float, keypoint_confidence_threshold: float, nms_threshold: float, nms_top_k: int, max_predictions: int):
        """
        :param score_threshold: Pose detection confidence threshold
        :param keypoint_confidence_threshold: A minimal confidence threshold for keypoints.
                                              Confidence scores of individual keypoints below this threshold will be set to 0.
        :param iou: IoU threshold for NMS step.
        :param nms_top_k: Number of predictions participating in NMS step
        :param max_predictions: maximum number of boxes to return after NMS step
        :param multi_label_per_box: controls whether to decode multiple labels per box.
                                    True - each anchor can produce multiple labels of different classes
                                           that pass confidence threshold check (default).
                                    False - each anchor can produce only one label of the class with the highest score.
        """
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.max_predictions = max_predictions
        self.keypoint_confidence_threshold = keypoint_confidence_threshold

    def __call__(self, outputs, device: str = None):
        """

        :param outputs:
        :param device:
        :return:
        """
        # First is model predictions, second element of tuple is logits for loss computation
        predictions = outputs[0]

        pred_poses = []
        pred_scores = []

        for pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores in zip(*predictions):
            # pred_bboxes [Anchors, 4] in XYXY format
            # pred_scores [Anchors, 1] confidence scores [0..1]
            # pred_pose_coords [Anchors, 17, 2] in (x,y) format
            # pred_pose_scores [Anchors, 17] confidence scores [0..1]

            pred_pose_scores[pred_pose_scores < self.keypoint_confidence_threshold] = 0.0

            pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
            conf_mask = pred_bboxes_conf >= self.score_threshold  # [Anchors]

            pred_bboxes_conf = pred_bboxes_conf[conf_mask]
            pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask]

            # Filter all predictions by self.nms_top_k
            if pred_bboxes_conf.size(0) > self.nms_top_k:
                topk_candidates = torch.topk(pred_bboxes_conf, k=self.nms_top_k, largest=True)
                pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
                pred_bboxes_xyxy = pred_bboxes_xyxy[topk_candidates.indices]
                pred_pose_coords = pred_pose_coords[topk_candidates.indices]
                pred_pose_scores = pred_pose_scores[topk_candidates.indices]

            # NMS
            idx_to_keep = torchvision.ops.boxes.nms(boxes=pred_bboxes_xyxy, scores=pred_bboxes_conf, iou_threshold=self.nms_threshold)
            idx_to_keep = idx_to_keep[: self.max_predictions]

            final_poses = torch.cat(
                [
                    pred_pose_coords[idx_to_keep],
                    pred_pose_scores[idx_to_keep].unsqueeze(-1),
                ],
                dim=-1,
            )  # [Instances, 17, 3]

            final_scores = pred_bboxes_conf[idx_to_keep]  # [Instances]

            pred_poses.append(final_poses)
            pred_scores.append(final_scores)

        return pred_poses, pred_scores
