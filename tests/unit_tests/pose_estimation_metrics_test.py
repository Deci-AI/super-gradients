import collections
import os.path
import tempfile
import unittest
from typing import List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json_tricks as json
from super_gradients.training.datasets.pose_estimation_datasets.coco_utils import remove_duplicate_annotations, make_keypoints_outside_image_invisible
from super_gradients.training.metrics.cocoeval import COCOeval as PatchedCOCOeval


class TestPoseEstimationMetrics(unittest.TestCase):
    def test_scores_match_with_pycocoeval(self):
        gt_annotations_path = "../data/coco2017/annotations/person_keypoints_val2017.json"
        assert os.path.isfile(gt_annotations_path)

        gt = COCO(gt_annotations_path)
        gt = remove_duplicate_annotations(gt)
        gt = make_keypoints_outside_image_invisible(gt)

        predictions = list(self.generate_noised_predictions(gt, instance_drop_probability=0.0, pose_offset=0.0))

        coco_pred = self.convert_predictions_to_coco_dict(predictions)

        with tempfile.TemporaryDirectory() as td:
            res_file = os.path.join(td, "keypoints_coco2017_results.json")

            with open(res_file, "w") as f:
                json.dump(coco_pred, f, sort_keys=True, indent=4)

            coco_dt = COCO(gt_annotations_path)
            coco_dt = coco_dt.loadRes(res_file)

        E = COCOeval(gt, coco_dt, iouType="keypoints")
        E.evaluate()  # run per image evaluation
        E.accumulate()  # accumulate per image results
        E.summarize()  # display summary metrics of results
        print(E.stats)

        E = PatchedCOCOeval(gt, coco_dt)
        E.evaluate()  # run per image evaluation
        E.accumulate()  # accumulate per image results
        E.summarize()  # display summary metrics of results
        print(E.stats)

    def generate_noised_predictions(self, coco: COCO, instance_drop_probability: float, pose_offset: float) -> List[Tuple[np.ndarray, int]]:
        """

        :param coco:
        :return: List of tuples (poses, image_id)
        """
        for image_id, image_info in coco.imgs.items():
            image_id_int = int(image_id)
            image_width = image_info["width"]
            image_height = image_info["height"]

            ann_ids = coco.getAnnIds(imgIds=image_id_int)
            anns = coco.loadAnns(ann_ids)

            poses = []
            for ann in anns:
                if np.random.rand() < instance_drop_probability:
                    continue

                keypoints = np.array(ann["keypoints"]).reshape(-1, 3).astype(np.float32)
                if pose_offset > 0:
                    keypoints[:, 0] += (2 * np.random.randn() - 1) * pose_offset
                    keypoints[:, 1] += (2 * np.random.randn() - 1) * pose_offset

                    keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_width)
                    keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_height)

                    # Apply random score for visible keypoints
                    keypoints[:, 2] = (keypoints[:, 2] > 0) * np.random.randn(len(keypoints))

                poses.append(keypoints)

            yield np.array(poses), image_id_int

    def convert_predictions_to_coco_dict(self, predictions):
        kpts = collections.defaultdict(list)
        for poses, image_id_int in predictions:
            # scores = np.random.rand(len(poses))
            scores = [1] * len(poses)
            for person_index, kpt in enumerate(poses):
                area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
                kpt = self._coco_process_keypoints(kpt)
                kpts[image_id_int].append({"keypoints": kpt[:, 0:3], "score": float(scores[person_index]), "image": image_id_int, "area": area})

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

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {"cat_id": self._class_to_coco_ind[cls], "cls_ind": cls_ind, "cls": cls, "ann_type": "keypoints", "keypoints": keypoints}
            for cls_ind, cls in enumerate(self.classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
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

    def _coco_process_keypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            # p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0) ## TODO: It was unused variable
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [float(keypoints[i][0]), float(keypoints[i][1]), float(keypoints[i][2])]

        return tmp


if __name__ == "__main__":
    unittest.main()
