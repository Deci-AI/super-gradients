import unittest
import numpy as np
import torch

from super_gradients.training.datasets import DetectionDataset
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat


class DummyDetectionDataset(DetectionDataset):
    def __init__(self, dataset_size, input_dim, *args, **kwargs):
        """Dummy Dataset testing subsampling."""

        self.dataset_size = dataset_size
        self.image_size = input_dim
        kwargs["all_classes_list"] = ["class_0", "class_1", "class_2"]
        kwargs["original_target_format"] = DetectionTargetsFormat.XYXY_LABEL
        super().__init__(data_dir="", input_dim=input_dim, *args, **kwargs)

    def _setup_data_source(self):
        return self.dataset_size

    def _load_annotation(self, sample_id: int) -> dict:
        """Load dummy annotation"""
        return {"img_path": "", "resized_img_shape": None, "target": torch.zeros(10, 6)}

    # DetectionDatasetV2 will call _load_image but since we don't have any image we patch this method with
    # tensor of image shape
    def _load_image(self, index: int) -> np.ndarray:
        return np.random.random(self.image_size)


class TestDetectionDatasetSubsampling(unittest.TestCase):
    def test_subsampling(self):
        """Check that subsampling works"""
        for max_num_samples in [1, 1_000, 1_000_000]:
            test_dataset = DummyDetectionDataset(dataset_size=100_000, input_dim=(640, 512), max_num_samples=max_num_samples)
            self.assertEqual(len(test_dataset), min(max_num_samples, 100_000))


if __name__ == "__main__":
    unittest.main()
