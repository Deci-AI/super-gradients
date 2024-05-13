import collections
import numbers
import typing
from typing import Dict, Optional, Union, Tuple, List

import cv2
import numpy as np
import super_gradients
import super_gradients.common.environment.ddp_utils
import torch
import torchvision.ops
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_metric
from super_gradients.module_interfaces.obb_predictions import OBBPredictions
from super_gradients.training.datasets.data_formats.obb.cxcywhr import cxcywhr_to_poly, poly_to_xyxy
from super_gradients.training.transforms.obb import OBBSample
from super_gradients.training.utils import tensor_container_to_device
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback, IouThreshold
from super_gradients.training.utils.detection_utils import (
    compute_detection_metrics,
    DetectionMatching,
    get_top_k_idx_per_cls,
)
from torch import Tensor
from torchmetrics import Metric

logger = get_logger(__name__)


class OBBIoUMatching(DetectionMatching):
    """
    IoUMatching is a subclass of DetectionMatching that uses Intersection over Union (IoU)
    for matching detections in object detection models.
    """

    def __init__(self, iou_thresholds: torch.Tensor):
        """
        Initializes the IoUMatching instance with IoU thresholds.

        :param iou_thresholds: (torch.Tensor) The IoU thresholds for matching.
        """
        self.iou_thresholds = iou_thresholds

    def get_thresholds(self) -> torch.Tensor:
        """
        Returns the IoU thresholds used for detection matching.

        :return: (torch.Tensor) The IoU thresholds.
        """
        return self.iou_thresholds

    @classmethod
    def pairwise_cxcywhr_iou_accurate(cls, obb1: Tensor, obb2: Tensor) -> Tensor:
        """
        Calculate the pairwise IoU between oriented bounding boxes.

        :param obb1: First set of boxes. Tensor of shape (N, 5) representing ground truth boxes, with cxcywhr format.
        :param obb2: Second set of boxes. Tensor of shape (M, 5) representing predicted boxes, with cxcywhr format.
        :return: A tensor of shape (N, M) representing IoU scores between corresponding boxes.
        """
        import numpy as np

        if len(obb1.shape) != 2 or len(obb2.shape) != 2:
            raise ValueError("Expected obb1 and obb2 to be 2D tensors")

        poly1 = cxcywhr_to_poly(obb1.detach().cpu().numpy())
        poly2 = cxcywhr_to_poly(obb2.detach().cpu().numpy())

        # Compute bounding boxes from polygons
        xyxy1 = poly_to_xyxy(poly1)
        xyxy2 = poly_to_xyxy(poly2)
        bbox_iou = torchvision.ops.box_iou(torch.from_numpy(xyxy1), torch.from_numpy(xyxy2)).numpy()
        iou = np.zeros((poly1.shape[0], poly2.shape[0]))

        # We use bounding box IoU to filter out pairs of polygons that has no intersection
        # Only polygons that have non-zero bounding box IoU are considered for polygon-polygon IoU calculation
        nz_indexes = np.nonzero(bbox_iou)
        for i, j in zip(*nz_indexes):
            iou[i, j] = cls.polygon_polygon_iou(poly1[i], poly2[j])
        return torch.from_numpy(iou).to(obb1.device)

    @classmethod
    def polygon_polygon_iou(cls, gt_rect, pred_rect):
        """
        Performs intersection over union calculation for two polygons using integer coordinates of
        vertices. This is a workaround for a bug in cv2.intersectConvexConvex function that returns
        incorrect results for polygons with float coordinates that are almost identical

        Args:
            gt_rect: [4,2]
            pred_rect: [4,2]

        Returns:

        """
        # Multiply by 1000 to account for rounding errors when going from float to int. 1000 should be enough to get rid of any rounding errors
        # It has no effect on IOU since it is scale-less
        pred_rect_int = (pred_rect * 1000).astype(int)
        gt_rect_int = (gt_rect * 1000).astype(int)

        try:
            intersection, _ = cv2.intersectConvexConvex(pred_rect_int, gt_rect_int, handleNested=True)
        except Exception as e:
            raise RuntimeError(
                "Detected error in cv2.intersectConvexConvex while calculating polygon_polygon_iou\n"
                f"pred_rect_int: {pred_rect_int}\n"
                f"gt_rect_int: {gt_rect_int}"
            ) from e

        gt_area = cv2.contourArea(gt_rect_int)
        pred_area = cv2.contourArea(pred_rect_int)

        # Second condition is to avoid division by zero when predicted polygon is degenerate (point or line)
        if intersection > 0 and pred_area > 0:
            union = gt_area + pred_area - intersection
            if union == 0:
                raise ZeroDivisionError(
                    f"ZeroDivisionError at polygon_polygon_iou_int\n"
                    f"Intersection is {intersection}\n"
                    f"Union is {union}\n"
                    f"gt_rect_int {gt_rect_int}\n"
                    f"pred_rect_int {pred_rect_int}"
                )
            return intersection / max(union, 1e-7)

        return 0

    def compute_targets(
        self,
        preds_cxcywhr: torch.Tensor,
        preds_cls: torch.Tensor,
        targets_cxcywhr: torch.Tensor,
        targets_cls: torch.Tensor,
        preds_matched: torch.Tensor,
        targets_matched: torch.Tensor,
        preds_idx_to_use: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the matching targets based on IoU for regular scenarios.

        :param preds_cxcywhr: (torch.Tensor) Predicted bounding boxes in CXCYWHR format.
        :param preds_cls: (torch.Tensor) Predicted classes.
        :param targets_cxcywhr: (torch.Tensor) Target bounding boxes in CXCYWHR format.
        :param targets_cls: (torch.Tensor) Target classes.
        :param preds_matched: (torch.Tensor) Tensor indicating which predictions are matched.
        :param targets_matched: (torch.Tensor) Tensor indicating which targets are matched.
        :param preds_idx_to_use: (torch.Tensor) Indices of predictions to use.
        :return: (torch.Tensor) Computed matching targets.
        """
        # shape = (n_preds x n_targets)
        iou = self.pairwise_cxcywhr_iou_accurate(preds_cxcywhr[preds_idx_to_use], targets_cxcywhr)

        # Fill IoU values at index (i, j) with 0 when the prediction (i) and target(j) are of different class
        # Filling with 0 is equivalent to ignore these values since with want IoU > iou_threshold > 0
        cls_mismatch = preds_cls[preds_idx_to_use].view(-1, 1) != targets_cls.view(1, -1)
        iou[cls_mismatch] = 0

        # The matching priority is first detection confidence and then IoU value.
        # The detection is already sorted by confidence in NMS, so here for each prediction we order the targets by iou.
        sorted_iou, target_sorted = iou.sort(descending=True, stable=True)

        # Only iterate over IoU values higher than min threshold to speed up the process
        for pred_selected_i, target_sorted_i in (sorted_iou > self.iou_thresholds[0]).nonzero(as_tuple=False):
            # pred_selected_i and target_sorted_i are relative to filters/sorting, so we extract their absolute indexes
            pred_i = preds_idx_to_use[pred_selected_i]
            target_i = target_sorted[pred_selected_i, target_sorted_i]

            # Vector[j], True when IoU(pred_i, target_i) is above the (j)th threshold
            is_iou_above_threshold = sorted_iou[pred_selected_i, target_sorted_i] > self.iou_thresholds

            # Vector[j], True when both pred_i and target_i are not matched yet for the (j)th threshold
            are_candidates_free = torch.logical_and(~preds_matched[pred_i, :], ~targets_matched[target_i, :])

            # Vector[j], True when (pred_i, target_i) can be matched for the (j)th threshold
            are_candidates_good = torch.logical_and(is_iou_above_threshold, are_candidates_free)

            # For every threshold (j) where target_i and pred_i can be matched together ( are_candidates_good[j]==True )
            # fill the matching placeholders with True
            targets_matched[target_i, are_candidates_good] = True
            preds_matched[pred_i, are_candidates_good] = True

            # When all the targets are matched with a prediction for every IoU Threshold, stop.
            if targets_matched.all():
                break

        return preds_matched

    def compute_crowd_targets(
        self,
        preds_cxcywhr: torch.Tensor,
        preds_cls: torch.Tensor,
        crowd_targets_cls: torch.Tensor,
        crowd_targets_cxcywhr: torch.Tensor,
        preds_matched: torch.Tensor,
        preds_to_ignore: torch.Tensor,
        preds_idx_to_use: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the matching targets based on IoU for crowd scenarios.

        :param preds_cxcywhr: (torch.Tensor) Predicted bounding boxes in CXCYWHR format.
        :param preds_cls: (torch.Tensor) Predicted classes.
        :param crowd_targets_cls: (torch.Tensor) Crowd target classes.
        :param crowd_targets_cxcywhr: (torch.Tensor) Crowd target bounding boxes in CXCYWHR format.
        :param preds_matched: (torch.Tensor) Tensor indicating which predictions are matched.
        :param preds_to_ignore: (torch.Tensor) Tensor indicating which predictions to ignore.
        :param preds_idx_to_use: (torch.Tensor) Indices of predictions to use.
        :return: (Tuple[torch.Tensor, torch.Tensor]) Computed matching targets for crowd scenarios.
        """
        # Crowd targets can be matched with many predictions.
        # Therefore, for every prediction we just need to check if it has IoU large enough with any crowd target.

        # shape = (n_preds_to_use x n_crowd_targets)
        iou = self.pairwise_cxcywhr_iou_accurate(preds_cxcywhr[preds_idx_to_use], crowd_targets_cxcywhr)

        # Fill IoA values at index (i, j) with 0 when the prediction (i) and target(j) are of different class
        # Filling with 0 is equivalent to ignore these values since with want IoA > threshold > 0
        cls_mismatch = preds_cls[preds_idx_to_use].view(-1, 1) != crowd_targets_cls.view(1, -1)
        iou[cls_mismatch] = 0

        # For each prediction, we keep it's highest score with any crowd target (of same class)
        # shape = (n_preds_to_use)
        best_ioa, _ = iou.max(1)

        # If a prediction has IoA higher than threshold (with any target of same class), then there is a match
        # shape = (n_preds_to_use x iou_thresholds)
        is_matching_with_crowd = best_ioa.view(-1, 1) > self.iou_thresholds.view(1, -1)

        preds_to_ignore[preds_idx_to_use] = torch.logical_or(preds_to_ignore[preds_idx_to_use], is_matching_with_crowd)

        return preds_matched, preds_to_ignore


def compute_obb_detection_matching(
    preds: OBBPredictions,
    targets: OBBSample,
    matching_strategy: OBBIoUMatching,
    top_k: Optional[int],
    output_device: Optional[torch.device] = None,
) -> Tuple:
    """
    Match predictions (NMS output) and the targets (ground truth) with respect to metric and confidence score
    for a given image.
    :param preds:           Tensor of shape (num_img_predictions, 6)
                            format:     (x1, y1, x2, y2, confidence, class_label) where x1,y1,x2,y2 are according to image size
    :param targets:         targets for this image of shape (num_img_targets, 6)
                            format:     (label, cx, cy, w, h) where cx,cy,w,h
    :param top_k:           Number of predictions to keep per class, ordered by confidence score
    :param matching_strategy: Method to match predictions to ground truth targets: IoU, distance based

    :return:
        :preds_matched:     Tensor of shape (num_img_predictions, n_thresholds)
                                True when prediction (i) is matched with a target with respect to the (j)th threshold
        :preds_to_ignore:   Tensor of shape (num_img_predictions, n_thresholds)
                                True when prediction (i) is matched with a crowd target with respect to the (j)th threshold
        :preds_scores:      Tensor of shape (num_img_predictions), confidence score for every prediction
        :preds_cls:         Tensor of shape (num_img_predictions), predicted class for every prediction
        :targets_cls:       Tensor of shape (num_img_targets), ground truth class for every target
    """
    num_thresholds = len(matching_strategy.get_thresholds())
    device = preds.scores.device
    num_preds = len(preds.rboxes_cxcywhr)

    targets_box = torch.from_numpy(targets.rboxes_cxcywhr[~targets.is_crowd]).to(device)
    targets_cls = torch.from_numpy(targets.labels[~targets.is_crowd]).to(device)

    crowd_target_box = torch.from_numpy(targets.rboxes_cxcywhr[targets.is_crowd]).to(device)
    crowd_targets_cls = torch.from_numpy(targets.labels[targets.is_crowd]).to(device)

    num_targets = len(targets_box)
    num_crowd_targets = len(crowd_target_box)

    if num_preds == 0:
        preds_matched = torch.zeros((0, num_thresholds), dtype=torch.bool, device=device)
        preds_to_ignore = torch.zeros((0, num_thresholds), dtype=torch.bool, device=device)
        preds_scores = torch.tensor([], dtype=torch.float32, device=device)
        preds_cls = torch.tensor([], dtype=torch.float32, device=device)
        targets_cls = targets_cls.to(device=device)
        return preds_matched, preds_to_ignore, preds_scores, preds_cls, targets_cls

    preds_scores = preds.scores
    preds_cls = preds.labels

    preds_matched = torch.zeros(num_preds, num_thresholds, dtype=torch.bool, device=device)
    targets_matched = torch.zeros(num_targets, num_thresholds, dtype=torch.bool, device=device)
    preds_to_ignore = torch.zeros(num_preds, num_thresholds, dtype=torch.bool, device=device)

    # Ignore all but the predictions that were top_k for their class
    if top_k is not None:
        preds_idx_to_use = get_top_k_idx_per_cls(preds_scores, preds_cls, top_k)
    else:
        preds_idx_to_use = torch.arange(num_preds, device=device)

    preds_to_ignore[:, :] = True
    preds_to_ignore[preds_idx_to_use] = False

    if num_targets > 0 or num_crowd_targets > 0:
        if num_targets > 0:
            preds_matched = matching_strategy.compute_targets(
                preds.rboxes_cxcywhr, preds_cls, targets_box, targets_cls, preds_matched, targets_matched, preds_idx_to_use
            )

        if num_crowd_targets > 0:
            preds_matched, preds_to_ignore = matching_strategy.compute_crowd_targets(
                preds.rboxes_cxcywhr, preds_cls, crowd_targets_cls, crowd_target_box, preds_matched, preds_to_ignore, preds_idx_to_use
            )

    if output_device is not None:
        preds_matched = preds_matched.to(output_device)
        preds_to_ignore = preds_to_ignore.to(output_device)
        preds_scores = preds_scores.to(output_device)
        preds_cls = preds_cls.to(output_device)
        targets_cls = targets_cls.to(output_device)

    return preds_matched, preds_to_ignore, preds_scores, preds_cls, targets_cls


@register_metric()
class OBBDetectionMetrics(Metric):
    """
    OBBDetectionMetrics

    Metric class for computing F1, Precision, Recall and Mean Average Precision.

    :param num_cls:                         Number of classes.
    :param post_prediction_callback:        DetectionPostPredictionCallback to be applied on net's output prior to the metric computation (NMS).
    :param iou_thres:                       IoU threshold to compute the mAP.
                                            Could be either instance of IouThreshold, a tuple (lower bound, upper_bound) or single scalar.
    :param recall_thres:                    Recall threshold to compute the mAP.
    :param score_thres:                     Score threshold to compute Recall, Precision and F1.
    :param top_k_predictions:               Number of predictions per class used to compute metrics, ordered by confidence score
    :param dist_sync_on_step:               Synchronize metric state across processes at each ``forward()`` before returning the value at the step.
    :param accumulate_on_cpu:               Run on CPU regardless of device used in other parts.
                                            This is to avoid "CUDA out of memory" that might happen on GPU.
    :param calc_best_score_thresholds       Whether to calculate the best score threshold overall and per class
                                            If True, the compute() function will return a metrics dictionary that not
                                            only includes the average metrics calculated across all classes,
                                            but also the optimal score threshold overall and for each individual class.
    :param include_classwise_ap:            Whether to include the class-wise average precision in the returned metrics dictionary.
                                            If enabled, output metrics dictionary will look similar to this:
                                            {
                                                'Precision0.5:0.95': 0.5,
                                                'Recall0.5:0.95': 0.5,
                                                'F10.5:0.95': 0.5,
                                                'mAP0.5:0.95': 0.5,
                                                'AP0.5:0.95_person': 0.5,
                                                'AP0.5:0.95_car': 0.5,
                                                'AP0.5:0.95_bicycle': 0.5,
                                                'AP0.5:0.95_motorcycle': 0.5,
                                                ...
                                            }
                                            Class names are either provided via the class_names parameter or are generated automatically.
    :param class_names:                     Array of class names. When include_classwise_ap=True, will use these names to make
                                            per-class APs keys in the output metrics dictionary.
                                            If None, will use dummy names `class_{idx}` instead.
    :param state_dict_prefix:               A prefix to append to the state dict of the metric. A state dict used to synchronize metric in DDP mode.
                                            It was empirically found that if you have two metric classes A and B(A) that has same state key, for
                                            some reason torchmetrics attempts to sync their states all toghether which causes an error.
                                            In this case adding a prefix to the name of the synchronized state seems to help,
                                            but it is still unclear why it happens.


    """

    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback,
        iou_thres: Tuple[float, ...],
        top_k_predictions: Optional[int] = None,
        recall_thres: Tuple[float, ...] = None,
        score_thres: Optional[float] = 0.01,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
        state_dict_prefix: str = "",
    ):
        if class_names is None:
            if include_classwise_ap:
                logger.warning(
                    "Parameter 'include_classwise_ap' is set to True, but no class names are provided. "
                    "We will generate dummy class names, but we recommend to provide class names explicitly to"
                    "have meaningful names in reported metrics."
                )
            class_names = ["class_" + str(i) for i in range(num_cls)]
        else:
            class_names = list(class_names)

        if class_names is not None and len(class_names) != num_cls:
            raise ValueError(f"Number of class names ({len(class_names)}) does not match number of classes ({num_cls})")

        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_cls = num_cls
        self.iou_thres = iou_thres
        self.class_names = class_names

        if isinstance(iou_thres, IouThreshold):
            self.iou_thresholds = iou_thres.to_tensor()
        elif isinstance(iou_thres, tuple):
            low, high = iou_thres
            self.iou_thresholds = IouThreshold.from_bounds(low, high)
        elif isinstance(iou_thres, typing.Iterable):
            self.iou_thresholds = torch.tensor(list(iou_thres)).float()
        elif isinstance(iou_thres, np.ndarray):
            self.iou_thresholds = torch.from_numpy(iou_thres).float()
        elif isinstance(iou_thres, numbers.Number):
            self.iou_thresholds = torch.tensor([iou_thres], dtype=torch.float32)

        self.map_str = "mAP" + self._get_range_str()
        self.include_classwise_ap = include_classwise_ap

        self.precision_metric_key = f"{state_dict_prefix}Precision{self._get_range_str()}"
        self.recall_metric_key = f"{state_dict_prefix}Recall{self._get_range_str()}"
        self.f1_metric_key = f"{state_dict_prefix}F1{self._get_range_str()}"
        self.map_metric_key = f"{state_dict_prefix}mAP{self._get_range_str()}"

        greater_component_is_better = [
            (self.precision_metric_key, True),
            (self.recall_metric_key, True),
            (self.map_metric_key, True),
            (self.f1_metric_key, True),
        ]

        if self.include_classwise_ap:
            self.per_class_ap_names = [f"{state_dict_prefix}AP{self._get_range_str()}_{class_name}" for class_name in class_names]
            greater_component_is_better += [(key, True) for key in self.per_class_ap_names]

        self.greater_component_is_better = collections.OrderedDict(greater_component_is_better)
        self.component_names = list(self.greater_component_is_better.keys())
        self.calc_best_score_thresholds = calc_best_score_thresholds
        self.best_threshold_per_class_names = [f"Best_score_threshold_{class_name}" for class_name in class_names]

        if self.calc_best_score_thresholds:
            self.component_names.append("Best_score_threshold")

        if self.calc_best_score_thresholds and self.include_classwise_ap:
            self.component_names += self.best_threshold_per_class_names

        self.components = len(self.component_names)

        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.world_size = None
        self.rank = None
        self.state_key = f"{state_dict_prefix}matching_info{self._get_range_str()}"
        self.add_state(self.state_key, default=[], dist_reduce_fx=None)

        self.recall_thresholds = torch.linspace(0, 1, 101) if recall_thres is None else torch.tensor(recall_thres, dtype=torch.float32)
        self.score_threshold = score_thres
        self.top_k_predictions = top_k_predictions

        self.accumulate_on_cpu = accumulate_on_cpu

    def update(self, preds, gt_samples: List[OBBSample]) -> None:
        """
        Apply NMS and match all the predictions and targets of a given batch, and update the metric state accordingly.

        :param preds:           Raw output of the model, the format might change from one model to another,
                                but has to fit the input format of the post_prediction_callback (cx,cy,wh)
        :param target:          Targets for all images of shape (total_num_targets, 6) LABEL_CXCYWH. format:  (index, label, cx, cy, w, h)
        :param device:          Device to run on
        :param inputs:          Input image tensor of shape (batch_size, n_img, height, width)
        :param crowd_targets:   Crowd targets for all images of shape (total_num_targets, 6), LABEL_CXCYWH
        """
        preds: List[OBBPredictions] = self.post_prediction_callback(preds)
        output_device = "cpu" if self.accumulate_on_cpu else None
        matching_strategy = OBBIoUMatching(self.iou_thresholds.to(preds[0].scores.device))

        for pred, trues in zip(preds, gt_samples):
            image_mathing = compute_obb_detection_matching(
                pred, trues, matching_strategy=matching_strategy, top_k=self.top_k_predictions, output_device=output_device
            )

            accumulated_matching_info = getattr(self, self.state_key)
            setattr(self, self.state_key, accumulated_matching_info + [image_mathing])

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute the metrics for all the accumulated results.
        :return: Metrics of interest
        """
        mean_ap, mean_precision, mean_recall, mean_f1, best_score_threshold = -1.0, -1.0, -1.0, -1.0, -1.0
        accumulated_matching_info = getattr(self, self.state_key)
        best_score_threshold_per_cls = np.zeros(self.num_cls)
        mean_ap_per_class = np.zeros(self.num_cls)

        if len(accumulated_matching_info):
            matching_info_tensors = [torch.cat(x, 0) for x in list(zip(*accumulated_matching_info))]

            # shape (n_class, nb_iou_thresh)
            (
                ap_per_present_classes,
                precision_per_present_classes,
                recall_per_present_classes,
                f1_per_present_classes,
                present_classes,
                best_score_threshold,
                best_score_thresholds_per_present_classes,
            ) = compute_detection_metrics(
                *matching_info_tensors,
                recall_thresholds=self.recall_thresholds,
                score_threshold=self.score_threshold,
                device="cpu" if self.accumulate_on_cpu else self.device,
            )

            # Precision, recall and f1 are computed for IoU threshold range, averaged over classes
            # results before version 3.0.4 (Dec 11 2022) were computed only for smallest value (i.e IoU 0.5 if metric is @0.5:0.95)
            mean_precision, mean_recall, mean_f1 = precision_per_present_classes.mean(), recall_per_present_classes.mean(), f1_per_present_classes.mean()

            # MaP is averaged over IoU thresholds and over classes
            mean_ap = ap_per_present_classes.mean()

            # Fill array of per-class AP scores with values for classes that were present in the dataset
            ap_per_class = ap_per_present_classes.mean(1)
            for i, class_index in enumerate(present_classes):
                mean_ap_per_class[class_index] = float(ap_per_class[i])
                best_score_threshold_per_cls[class_index] = float(best_score_thresholds_per_present_classes[i])

        output_dict = {
            self.precision_metric_key: float(mean_precision),
            self.recall_metric_key: float(mean_recall),
            self.map_metric_key: float(mean_ap),
            self.f1_metric_key: float(mean_f1),
        }

        if self.include_classwise_ap:
            for i, ap_i in enumerate(mean_ap_per_class):
                output_dict[self.per_class_ap_names[i]] = float(ap_i)

        if self.calc_best_score_thresholds:
            output_dict["Best_score_threshold"] = float(best_score_threshold)

        if self.include_classwise_ap and self.calc_best_score_thresholds:
            for threshold_per_class_names, threshold_value in zip(self.best_threshold_per_class_names, best_score_threshold_per_cls):
                output_dict[threshold_per_class_names] = float(threshold_value)

        return output_dict

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        """
        When in distributed mode, stats are aggregated after each forward pass to the metric state. Since these have all
        different sizes we override the synchronization function since it works only for tensors (and use
        all_gather_object)
        :param dist_sync_fn:
        :return:
        """
        if self.world_size is None:
            self.world_size = super_gradients.common.environment.ddp_utils.get_world_size() if self.is_distributed else -1
        if self.rank is None:
            self.rank = torch.distributed.get_rank() if self.is_distributed else -1

        if self.is_distributed:
            local_state_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
            gathered_state_dicts = [None] * self.world_size
            torch.distributed.barrier()
            torch.distributed.all_gather_object(gathered_state_dicts, local_state_dict)
            matching_info = []
            for state_dict in gathered_state_dicts:
                matching_info += state_dict[self.state_key]
            matching_info = tensor_container_to_device(matching_info, device="cpu" if self.accumulate_on_cpu else self.device)

            setattr(self, self.state_key, matching_info)

    def _get_range_str(self):
        return "@%.2f" % self.iou_thresholds[0] if not len(self.iou_thresholds) > 1 else "@%.2f:%.2f" % (self.iou_thresholds[0], self.iou_thresholds[-1])


@register_metric()
class OBBDetectionMetrics_050(OBBDetectionMetrics):
    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback,
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.01,
        top_k_predictions: Optional[int] = None,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
    ):
        super().__init__(
            num_cls=num_cls,
            post_prediction_callback=post_prediction_callback,
            iou_thres=IouThreshold.MAP_05,
            recall_thres=recall_thres,
            score_thres=score_thres,
            top_k_predictions=top_k_predictions,
            dist_sync_on_step=dist_sync_on_step,
            accumulate_on_cpu=accumulate_on_cpu,
            calc_best_score_thresholds=calc_best_score_thresholds,
            include_classwise_ap=include_classwise_ap,
            class_names=class_names,
            state_dict_prefix="",
        )


@register_metric()
class OBBDetectionMetrics_050_095(OBBDetectionMetrics):
    def __init__(
        self,
        num_cls: int,
        post_prediction_callback: DetectionPostPredictionCallback,
        recall_thres: torch.Tensor = None,
        score_thres: float = 0.01,
        top_k_predictions: Optional[int] = None,
        dist_sync_on_step: bool = False,
        accumulate_on_cpu: bool = True,
        calc_best_score_thresholds: bool = True,
        include_classwise_ap: bool = False,
        class_names: List[str] = None,
    ):
        super().__init__(
            num_cls=num_cls,
            post_prediction_callback=post_prediction_callback,
            iou_thres=IouThreshold.MAP_05_TO_095,
            recall_thres=recall_thres,
            score_thres=score_thres,
            top_k_predictions=top_k_predictions,
            dist_sync_on_step=dist_sync_on_step,
            accumulate_on_cpu=accumulate_on_cpu,
            calc_best_score_thresholds=calc_best_score_thresholds,
            include_classwise_ap=include_classwise_ap,
            class_names=class_names,
            state_dict_prefix="",
        )
