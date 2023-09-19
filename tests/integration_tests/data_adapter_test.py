import os
from copy import deepcopy
import numpy as np

from data_gradients.managers.detection_manager import DetectionAnalysisManager

from super_gradients import setup_device
from super_gradients.training.dataloaders.dataloaders import coco2017_val
from super_gradients.training.datasets.collate_fn import DetectionDatasetAdapterCollateFN
from super_gradients.training.datasets.collate_fn import BaseDatasetAdapterCollateFN


def test_python():
    # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
    loader = coco2017_val(dataset_params={"max_num_samples": 500, "with_crowd": False})  # `max_num_samples` To make it faster

    analyzer = DetectionAnalysisManager(
        report_title="TEST COCO Dataloader",
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


def test_from_dict():
    # We use Validation set because it does not include augmentation (which is random and makes it impossible to compare results)
    loader = coco2017_val(dataset_params={"max_num_samples": 500, "with_crowd": False})  # `max_num_samples` To make it faster

    analyzer = DetectionAnalysisManager(
        report_title="TEST COCO Dataloader",
        train_data=loader,
        val_data=loader,
        class_names=loader.dataset.classes,
        batches_early_stop=20,
        use_cache=True,  # With this we will be asked about the dataset information only once
        bbox_format="cxcywh",
        is_label_first=True,
    )
    os.remove(analyzer.config.cache_path)

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
                    "adapter_cache_path": "LOCAL.json",
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
    adapted_loader = DetectionDatasetAdapterCollateFN.adapt_dataloader(dataloader=deepcopy(loader), adapter_cache_path="local_path2.json", n_classes=80)

    for (adapted_images, adapted_targets), (images, targets) in zip(adapted_loader, loader):
        assert np.isclose(adapted_targets, targets).all()
        assert np.isclose(adapted_images, images).all()


if __name__ == "__main__":
    # test_python()
    # test_from_dict()
    test_ddp_from_dict_based_adapter()
    # test_ddp_python_based_adapter()
