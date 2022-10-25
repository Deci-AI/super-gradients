import unittest
import numpy as np
from pathlib import Path
import tempfile
import os

from super_gradients.training.datasets import DetectionDataset
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat


class DummyDetectionDataset(DetectionDataset):
    def __init__(self, input_dim, *args, **kwargs):
        """Dummy Dataset testing subclassing, designed with no annotation that includes class_2."""

        self.image_size = input_dim
        self.n_samples = 321
        kwargs["all_classes_list"] = ["class_0", "class_1", "class_2"]
        kwargs["original_target_format"] = DetectionTargetsFormat.XYXY_LABEL
        super().__init__(input_dim=input_dim, *args, **kwargs)

    def _setup_data_source(self):
        return self.n_samples

    def _load_annotation(self, sample_id: int) -> dict:
        """Every image is made of one target, with label sample_id % len(all_classes_list) and
        a seed to allow the random image to the same for a given sample_id
        """
        cls_id = sample_id % len(self.all_classes_list)
        return {"img_path": str(sample_id), "target": np.array([[0, 0, 10, 10, cls_id]]), "resized_img_shape": self.image_size, "seed": sample_id}

    # We overwrite this to fake images
    def _load_image(self, index: int) -> np.ndarray:
        np.random.seed(self.annotations[index]["seed"])  # Make sure that the generated random tensor of a given index will be the same over the runs
        return np.random.random((self.image_size[0], self.image_size[1], 3)) * 255


class TestDetectionDatasetCaching(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_cache_dir = tempfile.TemporaryDirectory(prefix="cache").name
        if not os.path.isdir(self.temp_cache_dir):
            os.mkdir(self.temp_cache_dir)

    def _count_cached_array(self):
        return len(list(Path(self.temp_cache_dir).glob("*.array")))

    def _empty_cache(self):
        for cache_file in Path(self.temp_cache_dir).glob("*.array"):
            cache_file.unlink()

    def test_cache_keep_empty(self):
        self._empty_cache()

        datasets = [
            DummyDetectionDataset(
                input_dim=(640, 512),
                ignore_empty_annotations=False,
                class_inclusion_list=class_inclusion_list,
                cache=True,
                cache_dir=self.temp_cache_dir,
                data_dir="/home/",
            )
            for class_inclusion_list in [["class_0", "class_1", "class_2"], ["class_0"], ["class_1"], ["class_2"], ["class_1", "class_2"]]
        ]

        self.assertEqual(1, self._count_cached_array())
        for first_dataset, second_dataset in zip(datasets[:-1], datasets[1:]):
            self.assertTrue(np.array_equal(first_dataset.cached_imgs_padded, second_dataset.cached_imgs_padded))

        self._empty_cache()

    def test_cache_ignore_empty(self):
        self._empty_cache()

        datasets = [
            DummyDetectionDataset(
                input_dim=(640, 512),
                ignore_empty_annotations=True,
                class_inclusion_list=class_inclusion_list,
                cache=True,
                cache_dir=self.temp_cache_dir,
                data_dir="/home/",
            )
            for class_inclusion_list in [["class_0", "class_1", "class_2"], ["class_0"], ["class_1"], ["class_2"], ["class_1", "class_2"]]
        ]

        self.assertEqual(5, self._count_cached_array())
        for first_dataset, second_dataset in zip(datasets[:-1], datasets[1:]):
            self.assertFalse(np.array_equal(first_dataset.cached_imgs_padded, second_dataset.cached_imgs_padded))

        self._empty_cache()

    def test_cache_saved(self):
        """Check that after the first time a dataset is called with specific params,
        the next time it will call the saved array instead of building it."""
        self._empty_cache()
        self.assertEqual(0, self._count_cached_array())

        _ = DummyDetectionDataset(input_dim=(640, 512), ignore_empty_annotations=True, cache=True, cache_dir=self.temp_cache_dir, data_dir="/home/")
        self.assertEqual(1, self._count_cached_array())

        for _ in range(5):
            _ = DummyDetectionDataset(input_dim=(640, 512), ignore_empty_annotations=True, cache=True, cache_dir=self.temp_cache_dir, data_dir="/home/")
            self.assertEqual(1, self._count_cached_array())

        self._empty_cache()


if __name__ == "__main__":
    unittest.main()
