import unittest
import numpy as np
from typing import List

from super_gradients.training.datasets import DetectionDataSetV2
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat

class TestDetectionDataSet(DetectionDataSetV2):

    def __init__(self, input_dim, *args, **kwargs):
        """Dummy Dataset testing subclassing, designed with no annotation that includes class_2."""
        self.image_size = input_dim
        kwargs['n_available_samples'] = 2
        kwargs['all_classes_list'] = ["class_0", "class_1", "class_2"]
        kwargs['target_format'] = DetectionTargetsFormat.XYXY_LABEL
        super().__init__(input_dim=input_dim, *args, **kwargs)

    def _load_annotation(self, sample_id: int) -> dict:
        """Load 2 different annotations.
            - Annotation 0 is made of: 3 targets of class 0, 0 of class_1 and 0 of class_2
            - Annotation 1 is made of: 2 targets of class_0, 1 of class_1 and 0 of class_2
        """
        return {"img_path": "", "target": _get_dummy_target(sample_id)}

    # We patch this to not have to have local image
    def _load_image(self, index: int) -> np.ndarray:
        return np.random.random(self.image_size)


def _get_dummy_target(sample_id: int) -> np.ndarray:
    """Return a dummy target depending on the sample_id"""
    if sample_id == 0:
        return np.array([[0, 0, 10, 10, 0],
                         [0, 5, 10, 15, 0],
                         [0, 5, 10, 15, 0]])
    if sample_id == 1:
        return np.array([[0, 0, 10, 10, 0],
                         [0, 5, 10, 15, 0],
                         [0, 5, 10, 15, 1]])


class TestDatasetInterface(unittest.TestCase):
    def setUp(self) -> None:
        self.CONFIG_KEEP_EMPTY_ANNOTATION = [
            {
                "class_inclusion_list": ["class_0", "class_1", "class_2"],
                "expected_n_targets_after_subclass": [3, 3]
            },
            {
                "class_inclusion_list": ["class_0"],
                "expected_n_targets_after_subclass": [3, 2]
            },
            {
                "class_inclusion_list": ["class_1"],
                "expected_n_targets_after_subclass": [0, 1]
            },
            {
                "class_inclusion_list": ["class_2"],
                "expected_n_targets_after_subclass": [0, 0]
            },
        ]
        self.CONFIG_IGNORE_EMPTY_ANNOTATION = [
            {
                "class_inclusion_list": ["class_0", "class_1", "class_2"],
                "expected_n_targets_after_subclass": [3, 3]
            },
            {
                "class_inclusion_list": ["class_0"],
                "expected_n_targets_after_subclass": [3, 2]
            },
            {
                "class_inclusion_list": ["class_1"],
                "expected_n_targets_after_subclass": [1]
            }
        ]

    def test_subclass_keep_empty(self):
        for config in self.CONFIG_KEEP_EMPTY_ANNOTATION:
            test_dataset = TestDetectionDataSet(input_dim=(640, 512), ignore_empty_annotations=False,
                                                class_inclusion_list=config["class_inclusion_list"])
            n_targets_after_subclass = _get_n_targets_after_subclass_per_index(test_dataset)
            self.assertListEqual(config["expected_n_targets_after_subclass"], n_targets_after_subclass)

    def test_subclass_drop_empty(self):
        for config in self.CONFIG_IGNORE_EMPTY_ANNOTATION:
            test_dataset = TestDetectionDataSet(input_dim=(640, 512), ignore_empty_annotations=True,
                                                class_inclusion_list=config["class_inclusion_list"])
            n_targets_after_subclass = _get_n_targets_after_subclass_per_index(test_dataset)
            self.assertListEqual(config["expected_n_targets_after_subclass"], n_targets_after_subclass)

        # Check last case when class_2, which should raise FileNotFoundError because not a single image has
        # a target in class_inclusion_list
        with self.assertRaises(FileNotFoundError):
            TestDetectionDataSet(input_dim=(640, 512), ignore_empty_annotations=True,
                                 class_inclusion_list=["class_2"])


def _get_n_targets_after_subclass_per_index(test_dataset: TestDetectionDataSet):
    """Iterate through every index of the dataset and return the associated number of targets per index"""
    dataset_target_len = []
    for index in range(len(test_dataset)):
        _img, targets = test_dataset[index]
        dataset_target_len.append(len(targets))
    return dataset_target_len


if __name__ == '__main__':
    unittest.main()
