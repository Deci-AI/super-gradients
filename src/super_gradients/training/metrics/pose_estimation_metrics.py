import collections
import itertools
import os
import tempfile
from typing import Dict, Union, List, Optional, Tuple, Callable, Iterable

import numpy as np
import pytorch_toolbelt.utils.distributed as ddp_toolbelt
import torch
from pytorch_toolbelt.utils import fs
from torch import Tensor
from torchmetrics import Metric

import super_gradients
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_metric
from super_gradients.training.metrics.pose_estimation_utils import compute_img_keypoint_matching

logger = get_logger(__name__)

__all__ = ["PoseEstimationMetrics"]


@register_metric("PoseEstimationMetrics")
class PoseEstimationMetrics(Metric):
    """ """

    def __init__(
        self,
        json_file: str,
        post_prediction_callback: Callable,
        num_joints: int,
        max_objects_per_image: int = 20,
        oks_sigmas: Optional[Iterable] = None,
        iou_thresholds: Optional[Iterable] = None,
        remove_duplicate_instances=False,
        remove_keypoints_outside_image=False,
    ):
        """

        :param json_file:
        :param post_prediction_callback:
        :param num_joints:
        :param oks_sigmas: OKS sigma factor for custom keypoint detection dataset
        """
        super().__init__(dist_sync_on_step=False)
        self.json_file = json_file
        self.num_joints = num_joints
        self.max_objects_per_image = max_objects_per_image
        self.remove_duplicate_instances = remove_duplicate_instances
        self.remove_keypoints_outside_image = remove_keypoints_outside_image
        self.stats_names = ["AP", "Ap .5", "AP .75", "AR", "AR .5", "AR .75"]
        self.greater_component_is_better = dict((k, True) for k in self.stats_names)

        self.oks_sigmas = None
        if oks_sigmas is not None:
            if len(oks_sigmas) != num_joints:
                raise ValueError("Length of oks_sigmas should be equal to num_joints")
            self.oks_sigmas = np.array(oks_sigmas).reshape(num_joints)
            logger.info(f"Using user-defined OKS sigmas {self.oks_sigmas}")

        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True, dtype=np.float32)

        self.iou_thresholds = torch.tensor(iou_thresholds)

        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)

        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.world_size = None
        self.rank = None
        self.add_state("predictions", default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        self.predictions = []

    @classmethod
    def mask_predictions_wrt_to_annotations(cls, predictions, target):
        gt_heatmap, mask, gt_offset, offset_weight = target

        # Apply masking to remove predictions on excluded areas
        heatmap, offset = predictions
        mask = mask.sum(dim=1, keepdim=True) > 0
        masked_heatmap = heatmap * mask
        masked_offset = offset * mask
        masked_preds = masked_heatmap, masked_offset
        return masked_preds

    def update(
        self,
        predictions: Tuple[Tensor, Tensor],
        target: torch.Tensor,
        gt_joints: List[np.ndarray],
        gt_areas: List[np.ndarray] = None,
        gt_bboxes: List[np.ndarray] = None,
    ):
        """
        Apply NMS and match all the predictions and targets of a given batch, and update the metric state accordingly.

        :param preds :        Raw output of the mode (heatmap, offsets)
        :param target:        Tuple of tensors (gt_heatmap, mask, gt_offset, offset_weight)

        :param gt_joints:        List of ground-truth joints for each image in the batch

        """
        masked_preds = self.mask_predictions_wrt_to_annotations(predictions, target)
        predictions = self.post_prediction_callback(masked_preds)  # Decode raw predictions into poses

        if gt_areas is None:
            gt_areas = [None] * len(gt_joints)

        if gt_bboxes is None:
            gt_bboxes = [None] * len(gt_joints)

        len(predictions)
        for i in range(len(predictions)):
            self.update_single_image(predictions[i], gt_joints[i], gt_areas=gt_areas[i], gt_bboxes=gt_bboxes[i])

    def update_single_image(self, predictions: Tuple[Tensor, Tensor], groundtruths: np.ndarray, gt_bboxes: Optional[Tensor], gt_areas: Optional[Tensor]):
        if len(predictions) == 0 and len(groundtruths) == 0:
            return

        pred_poses, pred_scores = predictions

        pred_poses = torch.tensor(pred_poses, dtype=torch.float, device=self.device)
        pred_scores = torch.tensor(pred_scores, dtype=torch.float, device=self.device)

        gt_keypoints = torch.tensor(groundtruths, dtype=torch.float, device=self.device)
        gt_is_ignore = torch.zeros_like(gt_keypoints[:, :, 2], dtype=torch.bool, device=self.device)  # TODO: Support is_crowd

        preds_matched, preds_to_ignore, preds_scores, num_targets = compute_img_keypoint_matching(
            pred_poses,
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
            iou_thresholds=self.iou_thresholds,
            sigmas=self.oks_sigmas,
            top_k=self.params.maxDets,
        )

        self.predictions.append((preds_matched, preds_to_ignore, preds_scores, num_targets))

    def _sync_dist(self, dist_sync_fn=None, process_group=None):
        """
        When in distributed mode, stats are aggregated after each forward pass to the metric state. Since these have all
        different sizes we override the synchronization function since it works only for tensors (and use
        all_gather_object)
        @param dist_sync_fn:
        @return:
        """
        gathered_predictions = ddp_toolbelt.all_gather(self.predictions)
        for node_id, p in enumerate(gathered_predictions):
            num_poses = sum([len(x[0]) for x in p])
            logger.info(f"Gathered {len(p)} predictions from node {node_id} with total poses {num_poses}")

        self.predictions = list(itertools.chain(*gathered_predictions))
        logger.info(f"Total predictions {len(self.predictions)}")

    def compute(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Compute the metrics for all the accumulated results.
        :return: Metrics of interest
        """
        predictions = self.predictions  # All gathered by this time
        total_poses = sum([len(x[0]) for x in predictions])
        logger.info(f"Total predictions {len(predictions)}, total poses {total_poses}")

        # A crutch to handle zero predictions since COCOEval cannot handle such cases
        if total_poses == 0:
            return collections.OrderedDict([(k, 0.0) for k in self.stats_names])

        with tempfile.TemporaryDirectory() as td:
            res_file = os.path.join(td, "keypoints_coco2017_results.json")

            # preds is a list of: image x person x (keypoints)
            # keypoints: num_joints * 4 (x, y, score, tag)
            kpts = collections.defaultdict(list)
            for predictions_index, (poses, scores, image_id_str) in enumerate(predictions):
                if len(poses) != len(scores):
                    raise RuntimeError("Number of poses does not match number of scores")

                for person_index, kpt in enumerate(poses):
                    area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                    kpt = self.processKeypoints(kpt)

                    image_id_int = int(fs.id_from_fname(image_id_str))

                    try:
                        kpts[image_id_int].append({"keypoints": kpt[:, 0:3], "score": float(scores[person_index]), "image": image_id_int, "area": area})
                    except Exception as e:
                        raise e

            # rescoring and oks nms
            oks_nmsed_kpts = []
            # image x person x (keypoints)
            for img in kpts.keys():
                # person x (keypoints)
                img_kpts = kpts[img]
                # person x (keypoints)
                # do not use nms, keep all detections
                keep = []
                if len(keep) == 0:
                    oks_nmsed_kpts.append(img_kpts)
                else:
                    oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

            self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)
            info_str = self._do_python_keypoint_eval(res_file)

        name_value = collections.OrderedDict(info_str)
        return name_value
