import dataclasses

import numpy as np
import torch
from torch import Tensor


def compute_visible_bbox_xywh(joints: Tensor, visibility_mask: Tensor) -> np.ndarray:
    """
    Compute the bounding box (X,Y,W,H) of the visible joints for each instance.

    :param joints:  [Num Instances, Num Joints, 2+] last channel must have dimension of
                    at least 2 that is considered to contain (X,Y) coordinates of the keypoint
    :param visibility_mask: [Num Instances, Num Joints]
    :return: A numpy array [Num Instances, 4] where last dimension contains bbox in format XYWH
    """
    visibility_mask = visibility_mask > 0
    initial_value = 1_000_000

    x1 = torch.min(joints[:, :, 0], where=visibility_mask, initial=initial_value, dim=-1)
    y1 = torch.min(joints[:, :, 1], where=visibility_mask, initial=initial_value, dim=-1)

    x1[x1 == initial_value] = 0
    y1[y1 == initial_value] = 0

    x2 = torch.max(joints[:, :, 0], where=visibility_mask, initial=0, dim=-1)
    y2 = torch.max(joints[:, :, 1], where=visibility_mask, initial=0, dim=-1)

    w = x2 - x1
    h = y2 - y1

    return torch.stack([x1, y1, w, h], dim=-1)


def compute_oks(
    pred_joints: Tensor,
    gt_joints: Tensor,
    gt_keypoint_visibility: Tensor,
    sigmas: Tensor,
    gt_areas: Tensor = None,
    gt_bboxes: Tensor = None,
) -> np.ndarray:
    """

    :param pred_joints: [K, NumJoints, 2] or [K, NumJoints, 3]
    :param pred_scores: [K]
    :param gt_joints:   [M, NumJoints, 2]
    :param gt_keypoint_visibility: [M, NumJoints]
    :param gt_areas: [M] Area of each ground truth instance. COCOEval uses area of the instance mask to scale OKs, so it must be provided separately.
        If None, we will use area of bounding box of each instance computed from gt_joints.

    :param gt_bboxes: [M, 4] Bounding box (X,Y,W,H) of each ground truth instance. If None, we will use bounding box of each instance computed from gt_joints.
    :param sigmas: [NumJoints]
    :return: IoU matrix [K, M]
    """

    ious = torch.zeros((len(pred_joints), len(gt_joints)), device=pred_joints.device)
    vars = (sigmas * 2) ** 2

    if gt_bboxes is None:
        gt_bboxes = compute_visible_bbox_xywh(gt_joints, gt_keypoint_visibility)

    if gt_areas is None:
        gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]

    # compute oks between each detection and ground truth object
    for gt_index, (gt_keypoints, gt_keypoint_visibility, gt_bbox, gt_area) in enumerate(zip(gt_joints, gt_keypoint_visibility, gt_bboxes, gt_areas)):
        # create bounds for ignore regions(double the gt bbox)
        xg = gt_keypoints[:, 0]
        yg = gt_keypoints[:, 1]
        k1 = torch.count_nonzero(gt_keypoint_visibility > 0)

        x0 = gt_bbox[0] - gt_bbox[2]
        x1 = gt_bbox[0] + gt_bbox[2] * 2
        y0 = gt_bbox[1] - gt_bbox[3]
        y1 = gt_bbox[1] + gt_bbox[3] * 2

        for pred_index, pred_keypoints in enumerate(pred_joints):
            xd = pred_keypoints[:, 0]
            yd = pred_keypoints[:, 1]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                dx = (x0 - xd).clamp_min(0) + (xd - x1).clamp_min(0)
                dy = (y0 - yd).clamp_min(0) + (yd - y1).clamp_min(0)

            e = (dx**2 + dy**2) / vars / (gt_area + torch.finfo(torch.float64).eps) / 2

            if k1 > 0:
                e = e[gt_keypoint_visibility > 0]
            ious[pred_index, gt_index] = torch.sum(torch.exp(-e)) / e.shape[0]

    return ious


@dataclasses.dataclass
class ImageKeypointMatchingResult:
    preds_matched: Tensor
    preds_to_ignore: Tensor
    preds_scores: Tensor
    num_targets: int


def compute_img_keypoint_matching(
    preds: Tensor,
    pred_scores: Tensor,
    targets: Tensor,
    targets_visibilities: Tensor,
    targets_areas: Tensor,
    targets_bboxes: Tensor,
    targets_ignored: Tensor,
    crowd_targets: Tensor,
    crowd_visibilities: Tensor,
    crowd_targets_areas: Tensor,
    crowd_targets_bboxes: Tensor,
    iou_thresholds: torch.Tensor,
    sigmas: Tensor,
    top_k: int,
) -> ImageKeypointMatchingResult:
    """
    Match predictions and the targets (ground truth) with respect to IoU and confidence score for a given image.

    :param preds:            Tensor of shape (K, NumJoints, 3) - Array of predicted skeletons.
                             Last dimension encode X,Y and confidence score of each joint

    :param pred_scores:      Tensor of shape (K) - Confidence scores for each pose

    :param targets:          Targets joints (M, NumJoints, 2) - Array of groundtruth skeletons

    :param targets_visibilities: Visibility status for each keypoint (M, NumJoints).
                             Values are 0 - invisible, 1 - occluded, 2 - fully visible

    :param targets_areas:    Tensor of shape (M) - Areas of target objects

    :param targets_bboxes:   Tensor of shape (M,4) - Bounding boxes (XYWH) of targets

    :param targets_ignored:  Tensor of shape (M) - Array of target that marked as ignored
                             (E.g all keypoints are not visible or target does not fit the desired area range)

    :param crowd_targets:    Targets joints (Mc, NumJoints, 3) - Array of groundtruth skeletons
                             Last dimension encode X,Y and visibility score of each joint:
                             (0 - invisible, 1 - occluded, 2 - fully visible)

    :param crowd_visibilities: Visibility status for each keypoint of crowd targets (Mc, NumJoints).
                             Values are 0 - invisible, 1 - occluded, 2 - fully visible

    :param crowd_targets_areas: Tensor of shape (Mc) - Areas of target objects

    :param crowd_targets_bboxes: Tensor of shape (Mc, 4) - Bounding boxes (XYWH) of crowd targets

    :param iou_thresholds:  IoU Threshold to compute the mAP

    :param sigmas:          Tensor of shape (NumJoints) with sigmas for each joint. Sigma value represent how 'hard'
                            it is to locate the exact groundtruth position of the joint.

    :param top_k:           Number of predictions to keep, ordered by confidence score

    :return:
        :preds_matched:     Tensor of shape (min(top_k, len(preds)), n_iou_thresholds)
                                True when prediction (i) is matched with a target with respect to the (j)th IoU threshold

        :preds_to_ignore:   Tensor of shape (min(top_k, len(preds)), n_iou_thresholds)
                                True when prediction (i) is matched with a crowd target with respect to the (j)th IoU threshold

        :preds_scores:      Tensor of shape (min(top_k, len(preds))) with scores of top-k predictions

        :num_targets:       Number of groundtruth targets (total num targets minus number of ignored)

    """
    num_iou_thresholds = len(iou_thresholds)

    device = preds.device if torch.is_tensor(preds) else (targets.device if torch.is_tensor(targets) else "cpu")
    num_targets = len(targets) - torch.count_nonzero(targets_ignored)

    preds_matched = torch.zeros(len(preds), num_iou_thresholds, dtype=torch.bool, device=device)
    targets_matched = torch.zeros(len(targets), num_iou_thresholds, dtype=torch.bool, device=device)
    preds_to_ignore = torch.zeros(len(preds), num_iou_thresholds, dtype=torch.bool, device=device)

    if preds is None or len(preds) == 0:
        return ImageKeypointMatchingResult(
            preds_matched=preds_matched,
            preds_to_ignore=preds_to_ignore,
            preds_scores=pred_scores,
            num_targets=num_targets.item(),
        )

    # Ignore all but the predictions that were top_k
    k = min(top_k, len(pred_scores))
    preds_idx_to_use = torch.topk(pred_scores, k=k, sorted=True, largest=True).indices
    preds_to_ignore[:, :] = True
    preds_to_ignore[preds_idx_to_use] = False

    if len(targets) > 0:
        iou = compute_oks(preds[preds_idx_to_use], targets, targets_visibilities, sigmas, gt_areas=targets_areas, gt_bboxes=targets_bboxes)

        # The matching priority is first detection confidence and then IoU value.
        # The detection is already sorted by confidence in NMS, so here for each prediction we order the targets by iou.
        sorted_iou, target_sorted = iou.sort(descending=True, stable=True)

        # Only iterate over IoU values higher than min threshold to speed up the process
        for pred_selected_i, target_sorted_i in (sorted_iou > iou_thresholds[0]).nonzero(as_tuple=False):

            # pred_selected_i and target_sorted_i are relative to filters/sorting, so we extract their absolute indexes
            pred_i = preds_idx_to_use[pred_selected_i]
            target_i = target_sorted[pred_selected_i, target_sorted_i]

            # Vector[j], True when IoU(pred_i, target_i) is above the (j)th threshold
            is_iou_above_threshold = sorted_iou[pred_selected_i, target_sorted_i] > iou_thresholds

            # Vector[j], True when both pred_i and target_i are not matched yet for the (j)th threshold
            are_candidates_free = torch.logical_and(~preds_matched[pred_i, :], ~targets_matched[target_i, :])

            # Vector[j], True when (pred_i, target_i) can be matched for the (j)th threshold
            are_candidates_good = torch.logical_and(is_iou_above_threshold, are_candidates_free)

            is_matching_with_ignore = are_candidates_free & are_candidates_good & targets_ignored[target_i]

            if preds_matched[pred_i].any() and is_matching_with_ignore.any():
                continue

            # For every threshold (j) where target_i and pred_i can be matched together ( are_candidates_good[j]==True )
            # fill the matching placeholders with True
            targets_matched[target_i, are_candidates_good] = True
            preds_matched[pred_i, are_candidates_good] = True

            preds_to_ignore[pred_i] = torch.logical_or(preds_to_ignore[pred_i], is_matching_with_ignore)

            # When all the targets are matched with a prediction for every IoU Threshold, stop.
            if targets_matched.all():
                break

    # Crowd targets can be matched with many predictions.
    # Therefore, for every prediction we just need to check if it has IoA large enough with any crowd target.
    if len(crowd_targets) > 0:
        # shape = (n_preds_to_use x n_crowd_targets)
        ioa = compute_oks(
            preds[preds_idx_to_use],
            crowd_targets,
            crowd_visibilities,
            sigmas,
            gt_areas=crowd_targets_areas,
            gt_bboxes=crowd_targets_bboxes,
        )

        # For each prediction, we keep it's highest score with any crowd target (of same class)
        # shape = (n_preds_to_use)
        best_ioa, _ = ioa.max(1)

        # If a prediction has IoA higher than threshold (with any target of same class), then there is a match
        # shape = (n_preds_to_use x iou_thresholds)
        is_matching_with_crowd = best_ioa.view(-1, 1) > iou_thresholds.view(1, -1)

        preds_to_ignore[preds_idx_to_use] = torch.logical_or(preds_to_ignore[preds_idx_to_use], is_matching_with_crowd)

    return ImageKeypointMatchingResult(
        preds_matched=preds_matched[preds_idx_to_use],
        preds_to_ignore=preds_to_ignore[preds_idx_to_use],
        preds_scores=pred_scores[preds_idx_to_use],
        num_targets=num_targets.item(),
    )
