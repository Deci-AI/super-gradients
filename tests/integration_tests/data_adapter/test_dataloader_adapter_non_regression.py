import os
import numpy as np
import unittest
import tempfile
import shutil

from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from data_gradients.managers.classification_manager import ClassificationAnalysisManager

from super_gradients.training.dataloaders.dataloaders import coco2017_val, cityscapes_stdc_seg50_val, cifar10_val
from super_gradients.training.dataloaders.adapters import (
    DetectionDataloaderAdapterFactory,
    SegmentationDataloaderAdapterFactory,
    ClassificationDataloaderAdapterFactory,
)


class DataloaderAdapterNonRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_adapter_on_coco2017_val(self):
        # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
        loader = coco2017_val(
            dataset_params={"max_num_samples": 500, "with_crowd": False},
            dataloader_params={"collate_fn": "DetectionCollateFN"},
        )  # `max_num_samples` To make it faster

        analyzer = DetectionAnalysisManager(
            report_title="coco2017_val",
            log_dir=self.tmp_dir,
            train_data=loader,
            val_data=loader,
            class_names=loader.dataset.classes,
            batches_early_stop=20,
            use_cache=True,  # With this we will be asked about the data information only once
            bbox_format="cxcywh",
            is_label_first=True,
        )
        analyzer.run()

        adapted_loader = DetectionDataloaderAdapterFactory.from_dataloader(dataloader=loader, config_path=analyzer.data_config.cache_path)

        for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
            assert np.isclose(adapted_targets, targets).all()
            assert np.isclose(adapted_images, images).all()
        os.remove(analyzer.data_config.cache_path)

    def test_adapter_on_cityscapes_stdc_seg50_val(self):
        # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
        loader = cityscapes_stdc_seg50_val()

        analyzer = SegmentationAnalysisManager(
            report_title="cityscapes_stdc_seg50_val",
            log_dir=self.tmp_dir,
            train_data=loader,
            val_data=loader,
            class_names=loader.dataset.classes + ["<unknown>"],
            batches_early_stop=1,
            use_cache=True,  # With this we will be asked about the data information only once
        )
        analyzer.run()

        adapted_loader = SegmentationDataloaderAdapterFactory.from_dataloader(dataloader=loader, config_path=analyzer.data_config.cache_path)

        for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
            assert np.isclose(adapted_targets, targets).all()
        os.remove(analyzer.data_config.cache_path)

    def test_adapter_on_cifar10_val(self):
        # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
        loader = cifar10_val(dataset_params={"transforms": ["ToTensor"]})

        analyzer = ClassificationAnalysisManager(
            report_title="test_python_classification",
            log_dir=self.tmp_dir,
            train_data=loader,
            val_data=loader,
            class_names=list(range(10)),
            batches_early_stop=20,
            use_cache=True,  # With this we will be asked about the data information only once
        )
        analyzer.run()

        adapted_loader = ClassificationDataloaderAdapterFactory.from_dataloader(dataloader=loader, config_path=analyzer.data_config.cache_path)

        for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
            assert np.isclose(adapted_targets, targets).all()
            assert np.isclose(adapted_images, images).all()
        os.remove(analyzer.data_config.cache_path)

    def test_adpter_from_dict(self):
        # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
        loader = coco2017_val(
            dataset_params={"max_num_samples": 500, "with_crowd": False},
            dataloader_params={"collate_fn": "DetectionCollateFN"},
        )  # `max_num_samples` To make it faster

        analyzer = DetectionAnalysisManager(
            report_title="coco2017_val_dict",
            log_dir=self.tmp_dir,
            train_data=loader,
            val_data=loader,
            class_names=loader.dataset.classes,
            batches_early_stop=20,
            use_cache=True,  # With this we will be asked about the data information only once
            bbox_format="cxcywh",
            is_label_first=True,
        )
        analyzer.run()

        # Here we mimic how it works when loading from a recipe
        adapted_loader = coco2017_val(
            dataset_params={"max_num_samples": 500, "with_crowd": False},
            dataloader_params={
                "collate_fn": {
                    "DetectionDatasetAdapterCollateFN": {
                        "base_collate_fn": "DetectionCollateFN",
                        "config_path": analyzer.data_config.cache_path,
                    }
                }
            },
        )

        for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
            assert np.isclose(adapted_targets, targets).all()
            assert np.isclose(adapted_images, images).all()
        os.remove(analyzer.data_config.cache_path)

    def test_ddp_from_dict_based_adapter(self):
        # setup_device(num_gpus=3)

        # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
        loader = coco2017_val(
            dataset_params={"max_num_samples": 500, "with_crowd": False},
            dataloader_params={"num_workers": 4, "collate_fn": "DetectionCollateFN"},
        )

        # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
        adapted_loader = coco2017_val(
            dataset_params={"max_num_samples": 500, "with_crowd": False},  # `max_num_samples` To make it faster
            dataloader_params={
                "num_workers": 4,
                "collate_fn": {
                    "DetectionDatasetAdapterCollateFN": {
                        "base_collate_fn": "DetectionCollateFN",
                        "config_path": os.path.join(self.tmp_dir, "test_ddp_from_dict_based_adapter.json"),
                    }
                },
            },
        )

        for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
            assert np.isclose(adapted_targets, targets).all()
            assert np.isclose(adapted_images, images).all()

    def test_ddp_python_based_adapter(self):
        # setup_device(num_gpus=3)

        # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
        loader = coco2017_val(
            dataset_params={"max_num_samples": 500, "with_crowd": False},  # `max_num_samples` To make it faster
            dataloader_params={"num_workers": 4, "collate_fn": "DetectionCollateFN"},
        )
        adapted_loader = DetectionDataloaderAdapterFactory.from_dataloader(
            dataloader=loader,
            config_path=os.path.join(self.tmp_dir, "test_ddp_python_based_adapter.json"),
        )

        for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
            assert np.isclose(adapted_targets, targets).all()
            assert np.isclose(adapted_images, images).all()


if __name__ == "__main__":
    DataloaderAdapterNonRegressionTest()
