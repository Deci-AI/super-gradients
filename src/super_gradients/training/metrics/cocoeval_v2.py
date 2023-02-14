import dataclasses
from collections import defaultdict
from typing import Union, List, Any, Tuple, Optional

import numpy as np
import torch
from pycocotools.coco import COCO
from torch import Tensor

from super_gradients.training.utils.detection_utils import compute_detection_metrics_per_cls


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


@dataclasses.dataclass
class EvaluationParams:
    """
    Params for computing pose estimation metrics
    """

    iou_thresholds: np.ndarray
    recall_thresholds: np.ndarray
    maxDets: int
    useCats: bool
    sigmas: np.ndarray

    @classmethod
    def get_predefined_coco_params(cls):
        """
        Create evaluation params for COCO dataset
        :return:
        """
        return cls(
            iou_thresholds=np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True, dtype=np.float32),
            recall_thresholds=np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True, dtype=np.float32),
            maxDets=20,
            useCats=True,
            sigmas=np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89], dtype=np.float32) / 10.0,
        )


@dataclasses.dataclass
class ImageLevelEvaluationResult:
    image_id: Union[str, int]
    category_id: int
    dtMatches: Any
    gtMatches: Any

    dtScores: List
    dtIgnore: Any

    gtIgnore: Any
    gtIsCrowd: np.ndarray


@dataclasses.dataclass
class DatasetLevelEvaluationResult:
    params: EvaluationParams
    precision: np.ndarray
    recall: np.ndarray

    @property
    def ap_metric(self):
        return self._summarize(1)

    @property
    def ar_metric(self):
        return self._summarize(1)

    def all_metrics(self):
        return {
            "AP": self._summarize(1),
            "AP_0.5": self._summarize(1, iouThr=0.5),
            "AP_0.75": self._summarize(1, iouThr=0.75),
            "AR": self._summarize(0),
            "AR_0.5": self._summarize(0, iouThr=0.5),
            "AR_0.75": self._summarize(0, iouThr=0.75),
        }

    def print(self):
        p = self.params

        def _print_summarize(ap=1, iouThr=None):
            score = self._summarize(ap, iouThr)
            iStr = " {:<18} {} @[ IoU={:<9} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = "{:0.2f}:{:0.2f}".format(p.iou_thresholds[0], p.iou_thresholds[-1]) if iouThr is None else "{:0.2f}".format(iouThr)
            print(iStr.format(titleStr, typeStr, iouStr, score))

        _print_summarize(1)
        _print_summarize(1, iouThr=0.5)
        _print_summarize(1, iouThr=0.75)
        _print_summarize(0)
        _print_summarize(0, iouThr=0.5)
        _print_summarize(0, iouThr=0.75)

    def _summarize(self, ap=1, iouThr=None):
        p = self.params

        if ap == 1:
            s = self.precision
        else:
            s = self.recall

        if iouThr is not None:
            t = np.where(iouThr == p.iou_thresholds)[0]
            s = s[t]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        return mean_s


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
        k1 = np.count_nonzero(gt_keypoint_visibility > 0)

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

            e = (dx**2 + dy**2) / vars / (gt_area + np.spacing(1)) / 2

            if k1 > 0:
                e = e[gt_keypoint_visibility > 0]
            ious[pred_index, gt_index] = torch.sum(torch.exp(-e)) / e.shape[0]

    return ious


class COCOevalV2:
    def __init__(self, params: EvaluationParams):
        self.params = params

    def evaluate_from_coco(self, groundtruth: COCO, predictions: COCO):
        """

        :param groundtruth: COCO-like object with ground truth annotations
        :param predictions: COCO-like object with predictions
        :return:
        """
        imgIds = list(sorted(groundtruth.getImgIds()))
        catIds = list(np.unique(groundtruth.getCatIds()))

        if self.params.useCats:
            gts = groundtruth.loadAnns(groundtruth.getAnnIds(imgIds=imgIds, catIds=catIds))
            dts = predictions.loadAnns(predictions.getAnnIds(imgIds=imgIds, catIds=catIds))
        else:
            gts = groundtruth.loadAnns(groundtruth.getAnnIds(imgIds=imgIds))
            dts = predictions.loadAnns(predictions.getAnnIds(imgIds=imgIds))

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            gt["ignore"] = (gt["num_keypoints"] == 0) or bool(gt["ignore"])

        _gts = defaultdict(list)  # gt for evaluation
        _dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            _gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            _dts[dt["image_id"], dt["category_id"]].append(dt)

        catIds = catIds if self.params.useCats else [-1]

        T = len(self.params.iou_thresholds)
        K = len(catIds)

        precision = -torch.ones((T, K))
        recall = -torch.ones((T, K))
        iou_thresholds = torch.from_numpy(self.params.iou_thresholds)
        recall_thresholds = torch.from_numpy(self.params.recall_thresholds)

        for k, catId in enumerate(catIds):

            eval_results = []
            for imgId in imgIds:
                groundtruths = _gts[imgId, catId]
                predictions = _dts[imgId, catId]

                if len(groundtruths) == 0 and len(predictions) == 0:
                    continue

                pred_keypoints = (
                    torch.stack([torch.tensor(pred["keypoints"], dtype=torch.float).reshape(-1, 3) for pred in predictions]) if len(predictions) else []
                )
                pred_scores = torch.tensor([pred["score"] for pred in predictions], dtype=torch.float)

                gt_keypoints = (
                    torch.stack([torch.tensor(gt["keypoints"], dtype=torch.float).reshape(-1, 3) for gt in groundtruths]) if len(groundtruths) else []
                )
                gt_areas = torch.tensor([gt["area"] for gt in groundtruths], dtype=torch.float)
                gt_bboxes = torch.tensor([gt["bbox"] for gt in groundtruths], dtype=torch.float)
                gt_is_ignore = torch.tensor([gt["ignore"] for gt in groundtruths], dtype=torch.bool)

                preds_matched, preds_to_ignore, preds_scores, num_targets = self.compute_img_keypoint_matching(
                    pred_keypoints,
                    pred_scores,
                    targets=gt_keypoints[~gt_is_ignore, :, 0:2] if len(groundtruths) else [],
                    targets_visibilities=gt_keypoints[~gt_is_ignore, :, 2] if len(groundtruths) else [],
                    targets_areas=gt_areas[~gt_is_ignore],
                    targets_bboxes=gt_bboxes[~gt_is_ignore],
                    targets_ignored=gt_is_ignore[~gt_is_ignore],
                    crowd_targets=gt_keypoints[gt_is_ignore, :, 0:2] if len(groundtruths) else [],
                    crowd_visibilities=gt_keypoints[gt_is_ignore, :, 2] if len(groundtruths) else [],
                    crowd_targets_areas=gt_areas[gt_is_ignore],
                    crowd_targets_bboxes=gt_bboxes[gt_is_ignore],
                    crowd_targets_ignored=gt_is_ignore[gt_is_ignore],
                    iou_thresholds=iou_thresholds,
                    top_k=self.params.maxDets,
                    imgId=imgId,
                )

                eval_results.append((preds_matched, preds_to_ignore, preds_scores, num_targets))

            preds_matched = torch.cat([x[0] for x in eval_results], dim=0)
            preds_to_ignore = torch.cat([x[1] for x in eval_results], dim=0)
            preds_scores = torch.cat([x[2] for x in eval_results], dim=0)
            n_targets = sum([x[3] for x in eval_results])

            if n_targets == 0:
                continue

            ap, _, cls_recall = compute_detection_metrics_per_cls(
                preds_matched=preds_matched,
                preds_to_ignore=preds_to_ignore,
                preds_scores=preds_scores,
                n_targets=n_targets,
                recall_thresholds=recall_thresholds,
                score_threshold=0,
                device="cpu",
            )
            precision[:, k] = ap
            recall[:, k] = cls_recall

        return DatasetLevelEvaluationResult(
            params=self.params,
            precision=precision.detach().cpu().numpy(),
            recall=recall.detach().cpu().numpy(),
        )

    def compute_img_keypoint_matching(
        self,
        preds: torch.Tensor,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        targets_visibilities: torch.Tensor,
        targets_areas: Optional[torch.Tensor],
        targets_bboxes: Optional[torch.Tensor],
        targets_ignored: Optional[torch.Tensor],
        crowd_targets: torch.Tensor,
        crowd_visibilities: torch.Tensor,
        crowd_targets_areas: Optional[torch.Tensor],
        crowd_targets_bboxes: Optional[torch.Tensor],
        crowd_targets_ignored: Optional[torch.Tensor],
        iou_thresholds: torch.Tensor,
        top_k: int,
        imgId: int,  # TODO: Remove me after debugging
    ) -> Tuple[Tensor, Tensor, Tensor, int]:
        """
        Match predictions and the targets (ground truth) with respect to IoU and confidence score for a given image.
        :param preds:           Tensor of shape (K, NumJoints, 3)
        :param targets:         targets for this image of shape (num_img_targets, 6)
                                format:     (index, x, y, w, h, label) where x,y,w,h are in range [0,1]
        :param crowd_targets:   crowd targets for all images of shape (total_num_crowd_targets, 6)
                                format:     (index, x, y, w, h, label) where x,y,w,h are in range [0,1]
        :param iou_thresholds:  Threshold to compute the mAP
        :param top_k:           Number of predictions to keep per class, ordered by confidence score

        :return:
            :preds_matched:     Tensor of shape (num_img_predictions, n_iou_thresholds)
                                    True when prediction (i) is matched with a target with respect to the (j)th IoU threshold
            :preds_to_ignore:   Tensor of shape (num_img_predictions, n_iou_thresholds)
                                    True when prediction (i) is matched with a crowd target with respect to the (j)th IoU threshold
        """
        num_iou_thresholds = len(iou_thresholds)

        device = preds.device if torch.is_tensor(preds) else (targets.device if torch.is_tensor(targets) else "cpu")

        if preds is None or len(preds) == 0:
            preds_matched = torch.zeros((0, num_iou_thresholds), dtype=torch.bool, device=device)
            preds_to_ignore = torch.zeros((0, num_iou_thresholds), dtype=torch.bool, device=device)
            preds_scores = torch.zeros((0,), dtype=torch.float, device=device)
            return preds_matched, preds_to_ignore, preds_scores, len(targets)

        preds_matched = torch.zeros(len(preds), num_iou_thresholds, dtype=torch.bool, device=device)
        targets_matched = torch.zeros(len(targets), num_iou_thresholds, dtype=torch.bool, device=device)
        preds_to_ignore = torch.zeros(len(preds), num_iou_thresholds, dtype=torch.bool, device=device)

        # Ignore all but the predictions that were top_k
        k = min(top_k, len(pred_scores))
        preds_idx_to_use = torch.topk(pred_scores, k=k, sorted=True, largest=True).indices
        preds_to_ignore[:, :] = True
        preds_to_ignore[preds_idx_to_use] = False

        sigmas = torch.from_numpy(self.params.sigmas).to(device)

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

        # return preds_matched, preds_to_ignore, pred_scores, len(targets)
        num_targets = len(targets) - torch.count_nonzero(targets_ignored)
        return preds_matched[preds_idx_to_use], preds_to_ignore[preds_idx_to_use], pred_scores[preds_idx_to_use], num_targets.item()
