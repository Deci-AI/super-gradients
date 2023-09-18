import unittest
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from data_gradients.managers.classification_manager import ClassificationAnalysisManager
from super_gradients.training.dataloaders.collate_fn import (
    DetectionDatasetAdapterCollateFN,
    SegmentationDatasetAdapterCollateFN,
    ClassificationDatasetAdapterCollateFN,
)


class SimpleDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def generate_masks(pattern_num):
    onehot_mask = torch.zeros((6, 640, 540), dtype=torch.uint8)

    if pattern_num == 0:
        onehot_mask[0, :213, :] = 1
        onehot_mask[1, 213:426, :] = 1
        onehot_mask[2, 426:, :] = 1
    elif pattern_num == 1:
        onehot_mask[3, :213, :] = 1
        onehot_mask[4, 213:426, :] = 1
        onehot_mask[5, 426:, :] = 1
    elif pattern_num == 2:
        for i in range(6):
            onehot_mask[i, i * 106 : (i + 1) * 106, i * 90 : (i + 1) * 90] = 1
    return onehot_mask


class TestDetectionAdapter(unittest.TestCase):
    def setUp(self) -> None:
        _source_targets = [
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            np.array([[10, 20, 10, 10, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            np.array([[30, 30, 10, 10, 2], [50, 50, 10, 10, 3], [0, 0, 0, 0, 0]]),
            np.array([[70, 70, 10, 10, 4], [80, 80, 10, 10, 5], [0, 0, 0, 0, 0]]),
            np.array([[50, 50, 10, 10, 2], [60, 70, 10, 10, 4], [30, 30, 10, 10, 1]]),
        ]
        _source_images = [np.random.randint(low=0, high=255, size=(3, 640, 540)) for _ in range(len(_source_targets))]
        self.dataset = SimpleDataset(images=_source_images, labels=_source_targets)

        # (N, 6) [sample_i, label, CX, CY, W, H)
        self.expected_targets_batches = [
            torch.tensor([[1, 1, 15, 25, 10, 10]]),
            torch.tensor([[0, 2, 35, 35, 10, 10], [0, 3, 55, 55, 10, 10], [1, 4, 75, 75, 10, 10], [1, 5, 85, 85, 10, 10]]),
            torch.tensor([[0, 2, 55, 55, 10, 10], [0, 4, 65, 75, 10, 10], [0, 1, 35, 35, 10, 10]]),
        ]
        self.expected_image_shapes_batches = [
            torch.Size([2, 3, 640, 540]),
            torch.Size([2, 3, 640, 540]),
            torch.Size([1, 3, 640, 540]),
        ]

    def test_adapting_dataset(self):

        analyzer_ds = DetectionAnalysisManager(
            report_title="TEST_DataAdapter_Dataset_Detection",
            train_data=self.dataset,
            val_data=self.dataset,
            class_names=list(map(str, range(6))),
            use_cache=True,
            is_label_first=False,
            bbox_format="xywh",
        )
        analyzer_ds.run()  # Run the analysis. This will create the cache.

        # We instantiate a torch dataloader using the adapter collate function
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=DetectionDatasetAdapterCollateFN(cache_path=analyzer_ds.config.cache_path))

        for expected_images_shape, expected_targets, (images, targets) in zip(self.expected_image_shapes_batches, self.expected_targets_batches, loader):
            self.assertEqual(images.shape, expected_images_shape)
            self.assertTrue(((0 <= images) & (images <= 1)).all())  # Should be 0-1
            self.assertTrue(torch.equal(targets, expected_targets))

    def test_adapting_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=2)
        analyzer_ds = DetectionAnalysisManager(
            report_title="TEST_DataAdapter_Dataloader_Detection",
            train_data=loader,
            val_data=loader,
            class_names=list(map(str, range(6))),
            use_cache=True,
            is_label_first=False,
            bbox_format="xywh",
        )
        analyzer_ds.run()  # Run the analysis. This will create the cache.

        # This is required to use the adapter inside the existing Dataloader.
        # `collate_fn=loader.collate_fn` ensure to still take into account any collate_fn that was passed to the Dataloader
        loader.collate_fn = DetectionDatasetAdapterCollateFN(collate_fn=loader.collate_fn, cache_path=analyzer_ds.config.cache_path)

        for expected_images_shape, expected_targets, (images, targets) in zip(self.expected_image_shapes_batches, self.expected_targets_batches, loader):
            self.assertEqual(images.shape, expected_images_shape)
            self.assertTrue(((0 <= images) & (images <= 1)).all())  # Should be 0-1
            self.assertTrue(torch.equal(targets, expected_targets))


class TestSegmentationAdapter(unittest.TestCase):
    def setUp(self) -> None:
        _source_masks_onehot = [generate_masks(i) for i in range(3)]
        _source_images = [np.random.randint(low=0, high=255, size=(3, 640, 540), dtype=np.uint8) for _ in range(len(_source_masks_onehot))]
        self.dataset = SimpleDataset(images=_source_images, labels=_source_masks_onehot)

        # Expected masks in categorical format
        self.expected_masks_batches = [
            torch.cat([_source_masks_onehot[0].argmax(0).unsqueeze(0), _source_masks_onehot[1].argmax(0).unsqueeze(0)], dim=0),
            torch.cat([_source_masks_onehot[2].argmax(0).unsqueeze(0)], dim=0),
        ]
        self.expected_image_shapes_batches = [
            torch.Size([2, 3, 640, 540]),
            torch.Size([1, 3, 640, 540]),
        ]

    def test_adapting_dataset(self):
        # Run the analysis on DATASET
        analyzer_ds = SegmentationAnalysisManager(
            report_title="TEST_DataAdapter_Dataset_Segmentation",
            train_data=self.dataset,
            val_data=self.dataset,
            class_names=list(map(str, range(6))),
            use_cache=True,
            is_batch=False,
        )
        analyzer_ds.run()

        # We instantiate a torch dataloader using the adapter collate function
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=SegmentationDatasetAdapterCollateFN(cache_path=analyzer_ds.config.cache_path))

        for expected_images_shape, expected_masks, (images, masks) in zip(self.expected_image_shapes_batches, self.expected_masks_batches, loader):
            self.assertEqual(images.shape, expected_images_shape)
            self.assertTrue((masks == expected_masks).all())  # Checking that the masks are as expected

    def test_adapting_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=2)

        # Run the analysis on DATALOADER
        analyzer_ds = SegmentationAnalysisManager(
            report_title="TEST_DataAdapter_Dataloader_Segmentation",
            train_data=loader,
            val_data=loader,
            class_names=list(map(str, range(6))),
            use_cache=True,
            is_batch=True,
        )
        analyzer_ds.run()

        # This is required to use the adapter inside the existing Dataloader.
        # `collate_fn=loader.collate_fn` ensure to still take into account any collate_fn that was passed to the Dataloader
        loader.collate_fn = SegmentationDatasetAdapterCollateFN(collate_fn=loader.collate_fn, cache_path=analyzer_ds.config.cache_path)

        for expected_images_shape, expected_masks, (images, masks) in zip(self.expected_image_shapes_batches, self.expected_masks_batches, loader):
            self.assertEqual(images.shape, expected_images_shape)
            self.assertTrue((masks == expected_masks).all())  # Checking that the masks are as expected


class TestClassificationAdapter(unittest.TestCase):
    def setUp(self) -> None:
        # 0 or 1 labels for this simple example
        _source_labels = np.array([0, 1, 0, 1, 0])
        _source_images = [np.random.randint(low=0, high=255, size=(3, 640, 540)) for _ in range(len(_source_labels))]
        self.dataset = SimpleDataset(images=_source_images, labels=_source_labels)

        self.expected_labels_batches = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0])]
        self.expected_image_shapes_batches = [
            torch.Size([2, 3, 640, 540]),
            torch.Size([2, 3, 640, 540]),
            torch.Size([1, 3, 640, 540]),
        ]

    def test_adapting_dataset(self):
        # Run the analysis on DATASET
        analyzer_ds = ClassificationAnalysisManager(
            report_title="TEST_DataAdapter_Dataset_Classification",
            train_data=self.dataset,
            val_data=self.dataset,
            class_names=list(map(str, range(6))),
            images_extractor="[0]",
            labels_extractor="[1]",
            use_cache=True,
            is_batch=False,
        )
        analyzer_ds.run()

        # We instantiate a torch dataloader using the adapter collate function
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=ClassificationDatasetAdapterCollateFN(cache_path=analyzer_ds.config.cache_path))

        for expected_images_shape, expected_labels, (images, labels) in zip(self.expected_image_shapes_batches, self.expected_labels_batches, loader):
            self.assertEqual(images.shape, expected_images_shape)
            self.assertTrue(torch.equal(labels, expected_labels))

    def test_adapting_dataloader(self):
        # We mimic a user instantiating a dataloader
        loader = DataLoader(self.dataset, batch_size=2)

        # Run the analysis on DATASET
        analyzer_ds = ClassificationAnalysisManager(
            report_title="TEST_DataAdapter_Dataset_Classification",
            train_data=loader,
            val_data=loader,
            class_names=list(map(str, range(6))),
            images_extractor="[0]",
            labels_extractor="[1]",
            use_cache=True,
            is_batch=True,
        )
        analyzer_ds.run()

        # This is required to use the adapter inside the existing Dataloader.
        # `collate_fn=loader.collate_fn` ensure to still take into account any collate_fn that was passed to the Dataloader
        loader.collate_fn = ClassificationDatasetAdapterCollateFN(collate_fn=loader.collate_fn, cache_path=analyzer_ds.config.cache_path)

        for expected_images_shape, expected_labels, (images, labels) in zip(self.expected_image_shapes_batches, self.expected_labels_batches, loader):
            self.assertEqual(images.shape, expected_images_shape)
            self.assertTrue(torch.equal(labels, expected_labels))


if __name__ == "__main__":
    unittest.main()
