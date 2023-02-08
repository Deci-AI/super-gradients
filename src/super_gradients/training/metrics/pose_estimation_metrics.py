import collections
import itertools
import os
import tempfile
import typing
from typing import Dict, Union, List, Optional

import json_tricks as json
import numpy as np
import pytorch_toolbelt.utils.distributed as ddp_toolbelt
import super_gradients
import torch
from pycocotools.coco import COCO
from pytorch_toolbelt.utils import fs
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry.registry import register_metric
from torch import Tensor
from torchmetrics import Metric

from super_gradients.training.metrics.patched_cocoeval import COCOeval
from pose_estimation.models.postprocessing import PoseEstimationPostPredictionCallback

logger = get_logger(__name__)

__all__ = ["PoseEstimationMetrics"]


class PoseEstimationMetricsV2(Metric):
    def __init__(
        self,
        post_prediction_callback: PoseEstimationPostPredictionCallback,
        num_joints: int,
        max_objects_per_image: int = 100,
        oks_sigmas: Optional[typing.Iterable] = None,
    ):
        super().__init__(dist_sync_on_step=False)
        self.num_joints = num_joints
        self.max_objects_per_image = max_objects_per_image
        self.stats_names = ["AP", "Ap .5", "AP .75", "AP (M)", "AP (L)", "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"]
        self.greater_component_is_better = dict((k, True) for k in self.stats_names)
        self.oks_sigmas = None
        if oks_sigmas is not None:
            if len(oks_sigmas) != num_joints:
                raise ValueError("Length of oks_sigmas should be equal to num_joints")
            self.oks_sigmas = np.array(oks_sigmas).reshape(num_joints)
            logger.info(f"Using user-defined OKS sigmas {self.oks_sigmas}")

        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)

        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.world_size = None
        self.rank = None
        # self.add_state("predictions", default=[], dist_reduce_fx=None)

    def update(self, preds: typing.Tuple[Tensor, Tensor], target: torch.Tensor, gt_joints: List[np.ndarray]):
        masked_preds = self.mask_predictions_wrt_to_annotations(preds, target)

        predictions = self.post_prediction_callback(masked_preds)  # Decode raw predictions into poses
        for (pred_poses, pred_scores), gt_poses in zip(predictions, gt_joints):
            result = self.match_poses(pred_poses, pred_scores, gt_poses)
            self.predictions += result


@register_metric("PoseEstimationMetrics")
class PoseEstimationMetrics(Metric):
    """

    Important notice: This metric expects that validation dataset does not resize images, and
    they come in original resolution
    """

    def __init__(
        self,
        json_file: str,
        post_prediction_callback: PoseEstimationPostPredictionCallback,
        num_joints: int,
        max_objects_per_image: int = 100,
        oks_sigmas: Optional[typing.Iterable] = None,
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
        self.stats_names = ["AP", "Ap .5", "AP .75", "AP (M)", "AP (L)", "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"]
        self.greater_component_is_better = dict((k, True) for k in self.stats_names)
        self.oks_sigmas = None
        if oks_sigmas is not None:
            if len(oks_sigmas) != num_joints:
                raise ValueError("Length of oks_sigmas should be equal to num_joints")
            self.oks_sigmas = np.array(oks_sigmas).reshape(num_joints)
            logger.info(f"Using user-defined OKS sigmas {self.oks_sigmas}")

        self.component_names = list(self.greater_component_is_better.keys())
        self.components = len(self.component_names)

        self.post_prediction_callback = post_prediction_callback
        self.is_distributed = super_gradients.is_distributed()
        self.world_size = None
        self.rank = None
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.classes = ["__background__", "person"]
        self._class_to_coco_ind = {"person": 1}

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

    def update(self, preds: typing.Tuple[Tensor, Tensor], target: torch.Tensor, inputs: torch.tensor, file_name: List[str], pose_scale_factor: List[float]):
        """
        Apply NMS and match all the predictions and targets of a given batch, and update the metric state accordingly.

        :param preds :        Raw output of the mode (heatmap, offsets)
        :param target:        Tuple of tensors (gt_heatmap, mask, gt_offset, offset_weight)
        :param inputs:        Input image tensor of shape (batch_size, n_img, height, width)

        :param file_name:        List of corresponding filenames. Used to match the predictions with ground-truth
        :param pose_scale_factor: Divide predicted joint coordinates by this scale factor to obtain the joints
                                  in the coordinate system of the ground-truth JSON.

        """
        masked_preds = self.mask_predictions_wrt_to_annotations(preds, target)

        predictions = self.post_prediction_callback(masked_preds)  # Decode raw predictions into poses
        for (poses, scores), file_name, scale in zip(predictions, file_name, pose_scale_factor):
            # Scale predictions back to resolution of the GT
            scaled_poses = poses.copy()
            if len(scaled_poses) > 0:
                scaled_poses[:, :, 0:2] /= scale

            self.predictions.append((scaled_poses, scores, file_name))  # Accumulate them in internal state

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

    def _do_python_keypoint_eval(self, res_file):
        coco = COCO(self.json_file)
        coco_dt = coco.loadRes(res_file)
        coco_eval = COCOeval(coco, coco_dt, "keypoints")
        coco_eval.params.useSegm = None
        if self.oks_sigmas is not None:
            coco_eval.params.sigmas = self.oks_sigmas
        coco_eval.params.maxDets = [self.max_objects_per_image]
        coco_eval.evaluate()
        coco_eval.accumulate_with_coco()
        coco_eval.summarize()
        stats_names = ["AP", "Ap .5", "AP .75", "AP (M)", "AP (L)", "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"]
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {"cat_id": self._class_to_coco_ind[cls], "cls_ind": cls_ind, "cls": cls, "ann_type": "keypoints", "keypoints": keypoints}
            for cls_ind, cls in enumerate(self.classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info("=> Writing results json to %s" % res_file)
        with open(res_file, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, "r") as f:
                for line in f:
                    content.append(line)
            content[-1] = "]"
            with open(res_file, "w") as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack["cat_id"]
        keypoints = data_pack["keypoints"]
        cat_results = []
        num_joints = self.num_joints

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]["keypoints"] for k in range(len(img_kpts))])
            key_points = np.zeros((_key_points.shape[0], num_joints * 3), dtype=np.float32)

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                # keypoints score.
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append(
                    {
                        "image_id": img_kpts[k]["image"],
                        "category_id": cat_id,
                        "keypoints": list(key_points[k]),
                        "score": img_kpts[k]["score"],
                        "bbox": list([left_top[0], left_top[1], w, h]),
                    }
                )

        return cat_results

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            # p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0) ## TODO: It was unused variable
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [float(keypoints[i][0]), float(keypoints[i][1]), float(keypoints[i][2])]

        return tmp
