import os
from copy import deepcopy
import numpy as np

from data_gradients.managers.detection_manager import DetectionAnalysisManager
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager
from data_gradients.managers.classification_manager import ClassificationAnalysisManager

from super_gradients import setup_device
from super_gradients.training.dataloaders.dataloaders import coco2017_val, cityscapes_stdc_seg50_val, cifar10_val
from super_gradients.training.datasets.collate_fn import (
    DetectionDatasetAdapterCollateFN,
    SegmentationDatasetAdapterCollateFN,
    ClassificationDatasetAdapterCollateFN,
)
from super_gradients.training.datasets.collate_fn import BaseDatasetAdapterCollateFN


def test_python_detection():
    # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
    loader = coco2017_val(dataset_params={"max_num_samples": 500, "with_crowd": False})  # `max_num_samples` To make it faster

    analyzer = DetectionAnalysisManager(
        report_title="coco2017_val",
        train_data=loader,
        val_data=loader,
        class_names=loader.dataset.classes,
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
        bbox_format="cxcywh",
        is_label_first=True,
    )
    analyzer.run()

    adapted_loader = DetectionDatasetAdapterCollateFN.adapt_dataloader(dataloader=deepcopy(loader), adapter_cache_path=analyzer.config.cache_path, n_classes=80)

    for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
        assert np.isclose(adapted_targets, targets).all()
        assert np.isclose(adapted_images, images).all()
    os.remove(analyzer.config.cache_path)


def test_python_segmentation():
    # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
    loader = cityscapes_stdc_seg50_val()

    analyzer = SegmentationAnalysisManager(
        report_title="cityscapes_stdc_seg50_val",
        train_data=loader,
        val_data=loader,
        class_names=loader.dataset.classes + ["<unknown>"],
        batches_early_stop=1,
        use_cache=True,  # With this we will be asked about the dataset information only once
    )
    analyzer.run()

    adapted_loader = SegmentationDatasetAdapterCollateFN.adapt_dataloader(
        dataloader=deepcopy(loader), adapter_cache_path=analyzer.config.cache_path, n_classes=20
    )

    for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
        assert np.isclose(adapted_targets, targets).all()
    os.remove(analyzer.config.cache_path)


def test_python_classification():
    # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
    loader = cifar10_val(dataset_params={"transforms": ["ToTensor"]})

    analyzer = ClassificationAnalysisManager(
        report_title="test_python_classification",
        train_data=loader,
        val_data=loader,
        class_names=list(range(10)),
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
    )
    analyzer.run()

    adapted_loader = ClassificationDatasetAdapterCollateFN.adapt_dataloader(
        dataloader=deepcopy(loader), adapter_cache_path=analyzer.config.cache_path, n_classes=80
    )

    for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
        assert np.isclose(adapted_targets, targets).all()
        assert np.isclose(adapted_images, images).all()
    os.remove(analyzer.config.cache_path)


def test_from_dict():
    # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
    loader = coco2017_val(dataset_params={"max_num_samples": 500, "with_crowd": False})  # `max_num_samples` To make it faster

    analyzer = DetectionAnalysisManager(
        report_title="coco2017_val_dict",
        train_data=loader,
        val_data=loader,
        class_names=loader.dataset.classes,
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
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
                    "collate_fn": "DetectionCollateFN",
                    "adapter_cache_path": analyzer.config.cache_path,
                    "n_classes": 80,
                }
            }
        },
    )

    for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
        assert np.isclose(adapted_targets, targets).all()
        assert np.isclose(adapted_images, images).all()
    os.remove(analyzer.config.cache_path)


def test_ddp_from_dict_based_adapter():
    setup_device(num_gpus=3)

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
                    "collate_fn": "DetectionCollateFN",
                    "adapter_cache_path": "LOCAL2.json",
                    "n_classes": 80,
                }
            },
        },
    )

    BaseDatasetAdapterCollateFN.calibrate_dataloader(adapted_loader)

    for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
        assert np.isclose(adapted_targets, targets).all()
        assert np.isclose(adapted_images, images).all()


def test_ddp_python_based_adapter():
    setup_device(num_gpus=3)

    # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
    loader = coco2017_val(
        dataset_params={"max_num_samples": 500, "with_crowd": False},  # `max_num_samples` To make it faster
        dataloader_params={"num_workers": 4, "collate_fn": "DetectionCollateFN"},
    )
    adapted_loader = DetectionDatasetAdapterCollateFN.adapt_dataloader(dataloader=deepcopy(loader), adapter_cache_path="xx.json", n_classes=80)

    for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
        assert np.isclose(adapted_targets, targets).all()
        assert np.isclose(adapted_images, images).all()


if __name__ == "__main__":
    test_python_detection()
    test_python_segmentation()
    test_python_classification()
    test_from_dict()
    # test_ddp_from_dict_based_adapter()
    # test_ddp_python_based_adapter()
