import os.path
import unittest

import numpy as np
from pycocotools.coco import COCO
from super_gradients.training.datasets.pose_estimation_datasets.coco_utils import parse_coco_into_keypoints_annotations, segmentation2mask


class COCOParsingTest(unittest.TestCase):
    """
    Unit test for checking whether our implementation of COCO parsing produce the same results as the original pycoctools implementation.
    """

    def setUp(self) -> None:
        self.data_dir = os.environ.get("SUPER_GRADIENTS_COCO_DATASET_DIR", "/data/coco")

        self.keypoint_annotations = [
            "annotations/person_keypoints_val2017.json",
            "annotations/person_keypoints_train2017.json",
        ]

    def test_keypoints_segmentation_masks(self):
        for annotation_file in self.keypoint_annotations:
            annotation_file = os.path.join(self.data_dir, annotation_file)
            with self.subTest(annotation_file=annotation_file):
                coco = COCO(annotation_file)

                global_intersection = 0.0
                global_cardinality = 0.0

                _, keypoints, annotations = parse_coco_into_keypoints_annotations(annotation_file, image_path_prefix=self.data_dir)
                num_keypoints = len(keypoints)

                self.assertEquals(len(annotations), len(coco.getImgIds()))

                for annotation in annotations:
                    img_id = annotation.image_id

                    img_metadata = coco.loadImgs([img_id])[0]
                    ann_ids = coco.getAnnIds(imgIds=[img_id])
                    anns = coco.loadAnns(ann_ids)

                    coco_areas = [ann["area"] for ann in anns]
                    coco_keypoints = np.array([np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3) for ann in anns]).reshape(-1, num_keypoints, 3)

                    self.assertArrayAlmostEqual(coco_areas, annotation.ann_areas, rtol=1e-5, atol=1)
                    self.assertArrayAlmostEqual(coco_keypoints, annotation.ann_keypoints, rtol=1e-5, atol=1)

                    for ann_index in range(len(anns)):
                        ann = anns[ann_index]
                        expected_mask = coco.annToMask(ann)
                        expected_mask[expected_mask > 0] = 1

                        actual_mask = segmentation2mask(annotation.ann_segmentations[ann_index], image_shape=(img_metadata["height"], img_metadata["width"]))
                        actual_mask[actual_mask > 0] = 1

                        global_intersection += np.sum(expected_mask * actual_mask, dtype=np.float64)
                        global_cardinality += np.sum(expected_mask + actual_mask, dtype=np.float64)

                        iou = np.sum(expected_mask * actual_mask) / (np.sum(expected_mask + actual_mask) - np.sum(expected_mask * actual_mask))

                        # Uncomment this to visualize the differences for low IoU scores (if it happens)
                        # if iou < 0.2:
                        #     cv2.imshow("expected", expected_mask * 255)
                        #     cv2.imshow("actual", actual_mask * 255)
                        #     cv2.imshow("diff", cv2.absdiff(expected_mask * 255, actual_mask * 255))
                        #     cv2.waitKey(0)
                        #     print(f"iou: {iou}")

                        self.assertGreater(iou, 0.2, msg=f"iou: {iou} for img_id: {img_id} ann_index: {ann_index}")

                global_iou = global_intersection / (global_cardinality - global_intersection)
                print(global_iou, annotation_file)
                # The polygon rasterization implementation in pycocotools is slightly different from the one we use (OpenCV)
                # To evaluate how well the masks are parsed, we calculate the global IoU between the all the masks instances
                # This is done intentionally to avoid the influece of the low IoU scores for extremely small masks
                self.assertGreater(global_iou, 0.98)

    def assertArrayAlmostEqual(self, first, second, rtol, atol):
        self.assertTrue(np.allclose(first, second, rtol=rtol, atol=atol))


if __name__ == "__main__":
    unittest.main()
