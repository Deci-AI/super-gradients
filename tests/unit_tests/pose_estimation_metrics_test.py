import collections
import os.path
import random
import tempfile
import unittest
from pprint import pprint
from typing import List, Tuple

import json_tricks as json
import numpy as np
import torch.cuda
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from super_gradients.training.datasets.pose_estimation_datasets.coco_utils import (
    remove_duplicate_annotations,
    make_keypoints_outside_image_invisible,
    remove_crowd_annotations,
)
from super_gradients.training.metrics.pose_estimation_metrics import PoseEstimationMetrics


class TestPoseEstimationMetrics(unittest.TestCase):
    def _load_coco_groundtruth(self, with_crowd: bool, with_duplicates: bool, with_invisible_keypoitns: bool):
        gt_annotations_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/coco2017/annotations/person_keypoints_val2017.json")
        assert os.path.isfile(gt_annotations_path)

        gt = COCO(gt_annotations_path)
        if not with_duplicates:
            gt = remove_duplicate_annotations(gt)

        if not with_invisible_keypoitns:
            gt = make_keypoints_outside_image_invisible(gt)

        if not with_crowd:
            gt = remove_crowd_annotations(gt)

        return gt

    def _internal_compare_method(self, with_crowd: bool, with_duplicates: bool, with_invisible_keypoitns: bool, device: str):
        random.seed(0)
        np.random.seed(0)

        # Load groundtruth annotations
        gt = self._load_coco_groundtruth(with_crowd, with_duplicates, with_invisible_keypoitns)

        # Generate predictions by randomly dropping some instances and adding noise to remaining poses
        (
            predicted_poses,
            predicted_scores,
            groundtruths_poses,
            groundtruths_iscrowd,
            groundtruths_areas,
            groundtruths_bboxes,
            image_ids,
        ) = self.generate_noised_predictions(gt, instance_drop_probability=0.1, pose_offset=1)

        # Compute metrics using SG implementation
        def convert_predictions_to_target_format(preds):
            # This is out predictions decode function. Here it's no-op since we pass decoded predictions as the input
            # but in real life this post-processing callback should be doing actual pose decoding & NMS
            return preds

        sg_metrics = PoseEstimationMetrics(
            post_prediction_callback=convert_predictions_to_target_format,
            num_joints=17,
            max_objects_per_image=20,
            iou_thresholds_to_report=(0.5, 0.75),
        ).to(device)

        sg_metrics.update(
            preds=(predicted_poses, predicted_scores),
            target=None,
            gt_joints=groundtruths_poses,
            gt_iscrowd=groundtruths_iscrowd,
            gt_areas=groundtruths_areas,
            gt_bboxes=groundtruths_bboxes,
        )

        actual_metrics = sg_metrics.compute()
        pprint(actual_metrics)

        coco_pred = self._coco_convert_predictions_to_dict(predicted_poses, predicted_scores, image_ids)

        with tempfile.TemporaryDirectory() as td:
            res_file = os.path.join(td, "keypoints_coco2017_results.json")

            with open(res_file, "w") as f:
                json.dump(coco_pred, f, sort_keys=True, indent=4)

            coco_dt = self._load_coco_groundtruth(with_crowd, with_duplicates, with_invisible_keypoitns)
            coco_dt = coco_dt.loadRes(res_file)

            coco_evaluator = COCOeval(gt, coco_dt, iouType="keypoints")
            coco_evaluator.evaluate()  # run per image evaluation
            coco_evaluator.accumulate()  # accumulate per image results
            coco_evaluator.summarize()  # display summary metrics of results
            expected_metrics = coco_evaluator.stats

        self.assertAlmostEquals(expected_metrics[0], actual_metrics["AP"], delta=0.002)
        self.assertAlmostEquals(expected_metrics[5], actual_metrics["AR"], delta=0.002)

    def test_compare_pycocotools_with_our_implementation_no_crowd(self):
        for device in ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]:
            self._internal_compare_method(False, True, True, device)

    def test_compare_pycocotools_with_our_implementation_no_duplicates(self):
        for device in ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]:
            self._internal_compare_method(True, False, True, device)

    def test_compare_pycocotools_with_our_implementation_no_invisible(self):
        for device in ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]:
            self._internal_compare_method(True, True, False, device)

    def test_metric_works_on_empty_predictions(self):
        # Compute metrics using SG implementation
        def convert_predictions_to_target_format(preds):
            # This is out predictions decode function. Here it's no-op since we pass decoded predictions as the input
            # but in real life this post-processing callback should be doing actual pose decoding & NMS
            return preds

        sg_metrics = PoseEstimationMetrics(
            post_prediction_callback=convert_predictions_to_target_format,
            num_joints=17,
            max_objects_per_image=20,
            iou_thresholds=None,
            oks_sigmas=None,
        )

        actual_metrics = sg_metrics.compute()
        pprint(actual_metrics)

        self.assertEqual(-1, actual_metrics["AP"])
        self.assertEqual(-1, actual_metrics["AR"])

    def generate_noised_predictions(self, coco: COCO, instance_drop_probability: float, pose_offset: float) -> Tuple[List, List, List]:
        """

        :param coco:
        :return: List of tuples (poses, image_id)
        """
        image_ids = []

        predicted_poses = []
        predicted_scores = []

        groundtruths_poses = []
        groundtruths_iscrowd = []
        groundtruths_areas = []
        groundtruths_bboxes = []

        for image_id, image_info in coco.imgs.items():
            image_id_int = int(image_id)
            image_width = image_info["width"]
            image_height = image_info["height"]

            ann_ids = coco.getAnnIds(imgIds=image_id_int)
            anns = coco.loadAnns(ann_ids)

            image_pred_keypoints = []
            image_gt_keypoints = []
            image_gt_iscrowd = []
            image_gt_areas = []
            image_gt_bboxes = []

            for ann in anns:
                gt_keypoints = np.array(ann["keypoints"]).reshape(-1, 3).astype(np.float32)

                image_gt_keypoints.append(gt_keypoints)
                image_gt_iscrowd.append(ann["iscrowd"])
                image_gt_areas.append(ann["area"])
                image_gt_bboxes.append(ann["bbox"])

                if np.random.rand() < instance_drop_probability:
                    continue

                keypoints = gt_keypoints.copy()
                if pose_offset > 0:
                    keypoints[:, 0] += (2 * np.random.randn() - 1) * pose_offset
                    keypoints[:, 1] += (2 * np.random.randn() - 1) * pose_offset

                    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_width)
                    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_height)

                    # Apply random score for visible keypoints
                    keypoints[:, 2] = (keypoints[:, 2] > 0) * np.random.randn(len(keypoints))

                image_pred_keypoints.append(keypoints)

            image_ids.append(image_id_int)
            predicted_poses.append(image_pred_keypoints)
            predicted_scores.append(np.random.rand(len(image_pred_keypoints)))

            groundtruths_poses.append(image_gt_keypoints)
            groundtruths_iscrowd.append(np.array(image_gt_iscrowd, dtype=bool))
            groundtruths_areas.append(np.array(image_gt_areas))
            groundtruths_bboxes.append(np.array(image_gt_bboxes))

        return predicted_poses, predicted_scores, groundtruths_poses, groundtruths_iscrowd, groundtruths_areas, groundtruths_bboxes, image_ids

    def _coco_convert_predictions_to_dict(self, predicted_poses, predicted_scores, image_ids):
        kpts = collections.defaultdict(list)
        for poses, scores, image_id_int in zip(predicted_poses, predicted_scores, image_ids):

            for person_index, kpt in enumerate(poses):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self._coco_process_keypoints(kpt)
                kpts[image_id_int].append({"keypoints": kpt[:, 0:3], "score": float(scores[person_index]), "image": image_id_int, "area": area})

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

        classes = ["__background__", "person"]
        _class_to_coco_ind = {cls: i for i, cls in enumerate(classes)}

        data_pack = [
            {"cat_id": _class_to_coco_ind[cls], "cls_ind": cls_ind, "cls": cls, "ann_type": "keypoints", "keypoints": oks_nmsed_kpts}
            for cls_ind, cls in enumerate(classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0], num_joints=17)
        return results

    def _coco_keypoint_results_one_category_kernel(self, data_pack, num_joints: int):
        cat_id = data_pack["cat_id"]
        keypoints = data_pack["keypoints"]
        cat_results = []

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

    def _coco_process_keypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [float(keypoints[i][0]), float(keypoints[i][1]), float(keypoints[i][2])]

        return tmp


if __name__ == "__main__":
    unittest.main()
