import itertools
from typing import Dict, Union, List, Optional, Tuple, Callable, Iterable, Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.ddp_utils import is_distributed
from super_gradients.common.object_names import Metrics
from super_gradients.common.registry.registry import register_metric
from super_gradients.training.metrics.pose_estimation_utils import compute_img_keypoint_matching, compute_visible_bbox_xywh
from super_gradients.training.utils import convert_to_tensor
from super_gradients.training.utils.detection_utils import compute_detection_metrics_per_cls

logger = get_logger(__name__)

__all__ = ["PoseEstimationMetrics"]


@register_metric(Metrics.POSE_ESTIMATION_METRICS)
class PoseEstimationMetrics(Metric):
    """
    Implementation of COCO Keypoint evaluation metric.
    When instantiated with default parameters, it will default to COCO params.
    By default, only AR and AP metrics are computed:

    >>> from super_gradients.training.metrics import PoseEstimationMetrics
    >>> metric = PoseEstimationMetrics(...)
    >>> metric.update(...)
    >>> metrics = metric.compute() # {"AP": 0.123, "AR": 0.456 }

    If you wish to get AR/AR at specific thresholds, you can specify them using `iou_thresholds_to_report` argument:

    >>> from super_gradients.training.metrics import PoseEstimationMetrics
    >>> metric = PoseEstimationMetrics(iou_thresholds_to_report=[0.5, 0.75], ...)
    >>> metric.update(...)
    >>> metrics = metric.compute() # {"AP": 0.123, "AP_0.5": 0.222, "AP_0.75: 0.111, "AR": 0.456, "AR_0.5":0.212, "AR_0.75": 0.443 }

    """

    def __init__(
        self,
        post_prediction_callback: Callable[[Any], Tuple[Tensor, Tensor]],
        num_joints: int,
        max_objects_per_image: int = 20,
        oks_sigmas: Optional[Iterable] = None,
        iou_thresholds: Optional[Iterable] = None,
        recall_thresholds: Optional[Iterable] = None,
        iou_thresholds_to_report: Optional[Iterable] = None,
    ):
        """
        Compute the AP & AR metrics for pose estimation. By default, this class returns only AP and AR values.
        If you need to get additional metrics (AP at specific threshold), pass these thresholds via `iou_thresholds_to_report` argument.

        :param post_prediction_callback:  A callback to decode model predictions to poses. This should be callable that takes input (model predictions)
                                          and returns a tuple of (poses, scores)

        :param num_joints:                Number of joints per pose

        :param max_objects_per_image:     Maximum number of predicted poses to include in evaluation (Top-K poses will be used).

        :param oks_sigmas:                OKS sigma factor for custom keypoint detection dataset.
                                          If None, then metric will use default OKS from COCO and expect num_joints to be equal 17

        :param recall_thresholds:         List of recall thresholds to compute AP.
                                          If None, then will use default 101 recall thresholds from COCO in range [0..1]

        :param iou_thresholds:            List of IoU thresholds to use. If None, then COCO version of IoU will be used (0.5 ... 0.95)

        :param: iou_thresholds_to_report: List of IoU thresholds to return in metric. By default, only AP/AR metrics are returned, but one
                                          may also request to return AP_0.5,AP_0.75,AR_0.5,AR_0.75 setting `iou_thresholds_to_report=[0.5, 0.75]`

        """
        super().__init__(dist_sync_on_step=False)
        self.num_joints = num_joints
        self.max_objects_per_image = max_objects_per_image
        self.stats_names = ["AP", "AR"]

        if recall_thresholds is None:
            recall_thresholds = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True, dtype=np.float32)
        self.recall_thresholds = torch.tensor(recall_thresholds, dtype=torch.float32)

        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True, dtype=np.float32)
        self.iou_thresholds = torch.tensor(iou_thresholds, dtype=torch.float32)

        if iou_thresholds_to_report is not None:
            self.iou_thresholds_to_report = np.array([float(t) for t in iou_thresholds_to_report], dtype=np.float32)

            if not np.isin(self.iou_thresholds_to_report, self.iou_thresholds).all():
                missing = ~np.isin(self.iou_thresholds_to_report, self.iou_thresholds)
                raise RuntimeError(
                    f"One or many IoU thresholds to report are not present in IoU thresholds. Missing thresholds: {self.iou_thresholds_to_report[missing]}"
                )

            self.stats_names += [f"AP_{t:.2f}" for t in self.iou_thresholds_to_report]
            self.stats_names += [f"AR_{t:.2f}" for t in self.iou_thresholds_to_report]
        else:
            self.iou_thresholds_to_report = None

        self.greater_component_is_better = dict((k, True) for k in self.stats_names)

        if oks_sigmas is None:
            oks_sigmas = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0

        if len(oks_sigmas) != num_joints:
            raise ValueError(f"Length of oks_sigmas ({len(oks_sigmas)}) should be equal to num_joints {num_joints}")

        self.oks_sigmas = torch.tensor(oks_sigmas).float()

        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)

        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = is_distributed()
        self.world_size = None
        self.rank = None
        self.add_state("predictions", default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        self.predictions.clear()

    @torch.no_grad()
    def update(
        self,
        preds,
        target,
        gt_joints: List[np.ndarray],
        gt_iscrowd: List[np.ndarray] = None,
        gt_bboxes: List[np.ndarray] = None,
        gt_areas: List[np.ndarray] = None,
    ):
        """
        Decode the predictions and update the metric

        :param preds :           Raw output of the model

        :param target:           Targets for the model training (rarely used for evaluation)

        :param gt_joints:        List of ground-truth joints for each image in the batch. Each element is a numpy array of shape (num_instances, num_joints, 3).
                                 Note that augmentation/preprocessing transformations (Affine transforms specifically) must also be applied to gt_joints.
                                 This is to ensure joint coordinates are transforms identically as image. This is differs form COCO evaluation,
                                 where predictions rescaled back to original size of the image.
                                 However, this makes code much more (unnecessary) complicated, so we do it differently and evaluate joints in the coordinate
                                 system of the predicted image.

        :param gt_iscrowd:       Optional argument indicating which instance is annotated with `iscrowd` flog and is not used for evaluation;
                                 If not provided, all instances are considered as non-crowd targets.
                                 For instance, in CrowdPose all instances are considered as "non-crowd".

        :param gt_bboxes:        Bounding boxes of the groundtruth instances (XYWH).
                                 This is COCO-specific and is used in OKS computation for instances w/o visible keypoints.
                                 If not provided, the bounding box is computed as the minimum bounding box that contains all visible keypoints.

        :param gt_areas:         Area of the groundtruth area. in COCO this is the area of the corresponding segmentation mask and not the bounding box,
                                 so it cannot be computed programmatically. It's value used in object-keypoint similarity metric (OKS) computation.
                                 If not provided, the area is computed as the product of the width and height of the bounding box.
                                 (For instance this is used in CrowdPose dataset)

        """
        predicted_poses, predicted_scores = self.post_prediction_callback(preds)  # Decode raw predictions into poses

        if gt_bboxes is None:
            gt_bboxes = [compute_visible_bbox_xywh(torch.tensor(joints[:, :, 0:2]), torch.tensor(joints[:, :, 2])) for joints in gt_joints]

        if gt_areas is None:
            gt_areas = [bboxes[:, 2] * bboxes[:, 3] for bboxes in gt_bboxes]

        if gt_iscrowd is None:
            gt_iscrowd = [[False] * len(x) for x in gt_joints]

        for i in range(len(predicted_poses)):
            self.update_single_image(
                predicted_poses[i], predicted_scores[i], gt_joints[i], gt_areas=gt_areas[i], gt_bboxes=gt_bboxes[i], gt_iscrowd=gt_iscrowd[i]
            )

    def update_single_image(
        self,
        predicted_poses: Union[Tensor, np.ndarray],
        predicted_scores: Union[Tensor, np.ndarray],
        groundtruths: Union[Tensor, np.ndarray],
        gt_bboxes: Union[Tensor, np.ndarray],
        gt_areas: Union[Tensor, np.ndarray],
        gt_iscrowd: Union[Tensor, np.ndarray, List[bool]],
    ):
        if len(predicted_poses) == 0 and len(groundtruths) == 0:
            return
        if len(predicted_poses) != len(predicted_scores):
            raise ValueError("Length of predicted poses and scores should be equal. Got {} and {}".format(len(predicted_poses), len(predicted_scores)))
        if len(groundtruths) != len(gt_areas) != len(gt_bboxes) != len(gt_iscrowd):
            raise ValueError(
                "Length of groundtruths, areas, bboxes and iscrowd should be equal. Got {} and {} and {} and {}".format(
                    len(groundtruths), len(gt_areas), len(gt_bboxes), len(gt_iscrowd)
                )
            )

        predicted_poses = convert_to_tensor(predicted_poses, dtype=torch.float32, device=self.device)
        predicted_scores = convert_to_tensor(predicted_scores, dtype=torch.float32, device=self.device)

        gt_keypoints = convert_to_tensor(groundtruths, dtype=torch.float32, device=self.device)
        gt_areas = convert_to_tensor(gt_areas, dtype=torch.float32, device=self.device)
        gt_bboxes = convert_to_tensor(gt_bboxes, dtype=torch.float32, device=self.device)
        gt_iscrowd = convert_to_tensor(gt_iscrowd, dtype=torch.bool, device=self.device)

        gt_keypoints_xy = gt_keypoints[:, :, 0:2]
        gt_keypoints_visibility = gt_keypoints[:, :, 2]
        gt_all_kpts_invisible = gt_keypoints_visibility.eq(0).all(dim=1)
        gt_is_ignore = gt_all_kpts_invisible | gt_iscrowd

        targets = gt_keypoints_xy[~gt_is_ignore] if len(groundtruths) else []
        targets_visibilities = gt_keypoints_visibility[~gt_is_ignore] if len(groundtruths) else []
        targets_areas = gt_areas[~gt_is_ignore] if len(groundtruths) else []
        targets_bboxes = gt_bboxes[~gt_is_ignore]
        targets_ignored = gt_is_ignore[~gt_is_ignore]

        crowd_targets = gt_keypoints_xy[gt_is_ignore] if len(groundtruths) else []
        crowd_visibilities = gt_keypoints_visibility[gt_is_ignore] if len(groundtruths) else []
        crowd_targets_areas = gt_areas[gt_is_ignore]
        crowd_targets_bboxes = gt_bboxes[gt_is_ignore]

        mr = compute_img_keypoint_matching(
            predicted_poses,
            predicted_scores,
            #
            targets=targets,
            targets_visibilities=targets_visibilities,
            targets_areas=targets_areas,
            targets_bboxes=targets_bboxes,
            targets_ignored=targets_ignored,
            #
            crowd_targets=crowd_targets,
            crowd_visibilities=crowd_visibilities,
            crowd_targets_areas=crowd_targets_areas,
            crowd_targets_bboxes=crowd_targets_bboxes,
            #
            iou_thresholds=self.iou_thresholds.to(self.device),
            sigmas=self.oks_sigmas.to(self.device),
            top_k=self.max_objects_per_image,
        )

        self.predictions.append((mr.preds_matched.cpu(), mr.preds_to_ignore.cpu(), mr.preds_scores.cpu(), int(mr.num_targets)))

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        """
        When in distributed mode, stats are aggregated after each forward pass to the metric state. Since these have all
        different sizes we override the synchronization function since it works only for tensors (and use
        all_gather_object)
        :param dist_sync_fn:
        :return:
        """
        if self.world_size is None:
            self.world_size = torch.distributed.get_world_size() if self.is_distributed else -1
        if self.rank is None:
            self.rank = torch.distributed.get_rank() if self.is_distributed else -1

        if self.is_distributed:
            local_state_dict = self.predictions
            gathered_state_dicts = [None] * self.world_size
            torch.distributed.all_gather_object(gathered_state_dicts, local_state_dict)
            self.predictions = list(itertools.chain(*gathered_state_dicts))

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute the metrics for all the accumulated results.
        :return: Metrics of interest
        """
        T = len(self.iou_thresholds)
        K = 1  # num categories

        precision = -np.ones((T, K))
        recall = -np.ones((T, K))

        predictions = self.predictions  # All gathered by this time
        if len(predictions) > 0:
            preds_matched = torch.cat([x[0].cpu() for x in predictions], dim=0)
            preds_to_ignore = torch.cat([x[1].cpu() for x in predictions], dim=0)
            preds_scores = torch.cat([x[2].cpu() for x in predictions], dim=0)
            n_targets = sum([x[3] for x in predictions])

            cls_precision, _, cls_recall = compute_detection_metrics_per_cls(
                preds_matched=preds_matched,
                preds_to_ignore=preds_to_ignore,
                preds_scores=preds_scores,
                n_targets=n_targets,
                recall_thresholds=self.recall_thresholds.cpu(),
                score_threshold=0,
                device="cpu",
            )

            precision[:, 0] = cls_precision.cpu().numpy()
            recall[:, 0] = cls_recall.cpu().numpy()

        def summarize(s):
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            return mean_s

        metrics = {"AP": summarize(precision), "AR": summarize(recall)}

        if self.iou_thresholds_to_report is not None and len(self.iou_thresholds_to_report):
            for t in self.iou_thresholds_to_report:
                mask = np.where(t == self.iou_thresholds)[0]
                metrics[f"AP_{t:.2f}"] = summarize(precision[mask])
                metrics[f"AR_{t:.2f}"] = summarize(recall[mask])

        return metrics
