import os.path
import unittest
import tempfile
import shutil
import torch

from torchvision.datasets import VOCDetection, VOCSegmentation, Caltech101
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode

from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from data_gradients.managers.classification_manager import ClassificationAnalysisManager
from data_gradients.dataset_adapters.config.data_config import SegmentationDataConfig
from data_gradients.utils.data_classes.image_channels import ImageChannels

from super_gradients.training.dataloaders.adapters import (
    DetectionDataloaderAdapterFactory,
    SegmentationDataloaderAdapterFactory,
    ClassificationDataloaderAdapterFactory,
)


class DataloaderAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        if os.getenv("DEBUG_DIR"):  # This is useful when debugging locally, to avoid downloading the dataset everytime
            self.tmp_dir = os.path.join(os.getenv("DEBUG_DIR"), "DataloaderAdapterNonRegressionTest")
        else:
            self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_torch_classification(self):
        class ToRGB:
            def __call__(self, pic):
                return pic.convert("RGB")

        train_set = Caltech101(root=self.tmp_dir, download=True, transform=Compose([ToRGB(), ToTensor(), Resize((512, 512))]))

        analyzer = ClassificationAnalysisManager(
            train_data=train_set,
            val_data=train_set,
            log_dir=self.tmp_dir,
            report_title="Caltech101",
            class_names=train_set.categories,
            image_channels=ImageChannels.from_str("RGB"),
            is_batch=False,
            labels_extractor="[1]",  # dataset returns (image, label)
            batches_early_stop=4,
            use_cache=True,
        )
        analyzer.run()

        train_loader = ClassificationDataloaderAdapterFactory.from_dataset(
            dataset=train_set,
            config_path=analyzer.data_config.cache_path,
            batch_size=20,
        )

        images, labels = next(iter(train_loader))
        self.assertTrue(images.shape == torch.Size([20, 3, 512, 512]))
        self.assertTrue(labels.shape == torch.Size([20]))

    def test_torchvision_detection(self):

        train_set = VOCDetection(
            root=self.tmp_dir,
            image_set="train",
            download=True,
            year="2007",
            transform=Compose([Resize(size=(720, 720))]),
        )
        val_set = VOCDetection(
            root=self.tmp_dir,
            image_set="val",
            download=True,
            year="2007",
            transform=Compose([Resize(size=(720, 720))]),
        )

        import numpy as np

        PASCAL_VOC_CLASS_NAMES = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )

        def voc_format_to_bbox(sample: tuple) -> np.ndarray:
            target_annotations = sample[1]
            targets = []
            for target in target_annotations["annotation"]["object"]:
                target_bbox = target["bndbox"]
                target_np = np.array(
                    [
                        PASCAL_VOC_CLASS_NAMES.index(target["name"]),
                        float(target_bbox["xmin"]),
                        float(target_bbox["ymin"]),
                        float(target_bbox["xmax"]),
                        float(target_bbox["ymax"]),
                    ]
                )
                targets.append(target_np)
            return np.array(targets, dtype=float)

        from data_gradients.dataset_adapters.config.data_config import DetectionDataConfig

        analyzer = DetectionAnalysisManager(
            report_title="VOC_from_torch",
            log_dir=self.tmp_dir,
            train_data=train_set,
            val_data=val_set,
            labels_extractor=voc_format_to_bbox,
            class_names=list(PASCAL_VOC_CLASS_NAMES),
            image_channels=ImageChannels.from_str("RGB"),
            # class_names=train_set,
            batches_early_stop=20,
            use_cache=True,  # With this we will be asked about the dataset information only once
            is_label_first=True,
            bbox_format="cxcywh",
        )

        analyzer.run()
        config = DetectionDataConfig(labels_extractor=voc_format_to_bbox, cache_path=analyzer.data_config.cache_path)
        train_loader = DetectionDataloaderAdapterFactory.from_dataset(
            dataset=train_set,
            config=config,
            batch_size=20,
            num_workers=0,
            drop_last=True,
        )
        val_loader = DetectionDataloaderAdapterFactory.from_dataset(
            dataset=train_set,
            config=config,
            batch_size=20,
            num_workers=0,
            drop_last=True,
        )

        for images, labels in train_loader:
            self.assertTrue(images.ndim == 4)
            self.assertTrue(images.shape[:2] == torch.Size([20, 3]))
            self.assertTrue(labels.ndim == 2)
            self.assertTrue(labels.shape[-1] == 6)

        for images, labels in val_loader:
            self.assertTrue(images.ndim == 4)
            self.assertTrue(images.shape[:2] == torch.Size([20, 3]))
            self.assertTrue(labels.ndim == 2)
            self.assertTrue(labels.shape[-1] == 6)

    def test_torchvision_segmentation(self):
        train_set = VOCSegmentation(
            root=self.tmp_dir,
            image_set="train",
            download=True,
            year="2007",
            transform=Compose([Resize(size=(720, 720))]),
            target_transform=Compose([Resize((720, 720), interpolation=InterpolationMode.NEAREST)]),
        )
        val_set = VOCSegmentation(
            root=self.tmp_dir,
            image_set="val",
            download=True,
            year="2007",
            transform=Compose([Resize(size=(720, 720))]),
            target_transform=Compose([Resize((720, 720), interpolation=InterpolationMode.NEAREST)]),
        )

        analyzer = SegmentationAnalysisManager(
            report_title="VOC_SEG_from_torch2",
            log_dir=self.tmp_dir,
            train_data=train_set,
            val_data=val_set,
            class_names=[f"class_{i}" for i in range(256)],
            image_channels=ImageChannels.from_str("RGB"),
            # class_names=train_set,
            batches_early_stop=20,
            use_cache=True,  # With this we will be asked about the dataset information only once
        )

        analyzer.run()
        config = SegmentationDataConfig(cache_path=analyzer.data_config.cache_path)
        train_loader = SegmentationDataloaderAdapterFactory.from_dataset(
            dataset=train_set,
            config=config,
            batch_size=20,
            num_workers=0,
            drop_last=True,
        )
        val_loader = SegmentationDataloaderAdapterFactory.from_dataset(
            dataset=train_set,
            config=config,
            batch_size=20,
            num_workers=0,
            drop_last=True,
        )

        for images, labels in train_loader:
            self.assertTrue(images.shape == torch.Size([20, 3, 720, 720]))
            self.assertTrue(labels.shape == torch.Size([20, 720, 720]))

        for images, labels in val_loader:
            self.assertTrue(images.shape == torch.Size([20, 3, 720, 720]))
            self.assertTrue(labels.shape == torch.Size([20, 720, 720]))


if __name__ == "__main__":
    DataloaderAdapterTest()
