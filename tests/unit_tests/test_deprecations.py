import unittest
import warnings
from typing import Union

from omegaconf import DictConfig
from torch import nn

from super_gradients import setup_device, Trainer
from super_gradients.common.registry import register_model
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import CustomizableDetector, get_arch_params, ResNet18
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils.utils import arch_params_deprecated
from super_gradients.training.transforms.transforms import DetectionTargetsFormatTransform, DetectionHorizontalFlip, DetectionPaddedRescale


@register_model("DummyModel")
class DummyModel(CustomizableDetector):
    def __init__(self, arch_params: Union[str, dict, HpmStruct, DictConfig]):
        super().__init__(arch_params)


@register_model("DummyModelV2")
class DummyModelV2(nn.Module):
    @arch_params_deprecated
    def __init__(self, backbone, head, neck):
        super().__init__()


class DeprecationsUnitTest(unittest.TestCase):
    def test_deprecated_arch_params_inside_base_class_via_direct_call(self):
        arch_params = get_arch_params("yolo_nas_l_arch_params")
        arch_params = HpmStruct(**arch_params)
        model = DummyModel(arch_params)
        assert isinstance(model, DummyModel)

    def test_deprecated_arch_params_inside_base_class_via_models_get(self):
        arch_params = get_arch_params("yolo_nas_l_arch_params")
        model = models.get("DummyModel", arch_params=arch_params, num_classes=80)
        assert isinstance(model, DummyModel)

    def test_deprecated_arch_params_top_level_class_via_direct_call(self):
        arch_params = HpmStruct(backbone=dict(), head=dict(), neck=dict())
        model = DummyModelV2(arch_params)
        assert isinstance(model, DummyModelV2)

    def test_deprecated_arch_params_top_level_class_via_models_get(self):
        arch_params = dict(backbone=dict(), head=dict(), neck=dict())
        model = models.get("DummyModelV2", arch_params=arch_params, num_classes=80)
        assert isinstance(model, DummyModelV2)

    def test_deprecated_make_divisible(self):
        try:
            with self.assertWarns(DeprecationWarning):
                from super_gradients.training.models import make_divisible  # noqa

                assert make_divisible(1, 1) == 1
        except ImportError:
            self.fail("ImportError raised unexpectedly for make_divisible")

    def test_deprecated_BasicBlock(self):
        try:
            with self.assertWarns(DeprecationWarning):
                from super_gradients.training.models import BasicBlock, BasicResNetBlock  # noqa

                assert isinstance(BasicBlock(1, 1, 1), BasicResNetBlock)
        except ImportError:
            self.fail("ImportError raised unexpectedly for BasicBlock")

    def test_deprecated_max_targets(self):
        with self.assertWarns(DeprecationWarning):
            DetectionTargetsFormatTransform(max_targets=1)
            DetectionHorizontalFlip(prob=1.0, max_targets=1)
            DetectionPaddedRescale(input_dim=(2, 2), max_targets=1)

    def test_moved_Bottleneck_import(self):
        try:
            with self.assertWarns(DeprecationWarning):
                from super_gradients.training.models import Bottleneck as OldBottleneck  # noqa
                from super_gradients.training.models.classification_models.resnet import Bottleneck

                assert isinstance(OldBottleneck(1, 1, 1), Bottleneck)
        except ImportError:
            self.fail("ImportError raised unexpectedly for Bottleneck")

    def test_deprecated_optimizers_dict(self):
        try:
            with self.assertWarns(DeprecationWarning):
                from super_gradients.training.utils.optimizers.all_optimizers import OPTIMIZERS  # noqa
        except ImportError:
            self.fail("ImportError raised unexpectedly for OPTIMIZERS")

    def test_deprecated_HpmStruct_import(self):
        try:
            with self.assertWarns(DeprecationWarning):
                from super_gradients.training.models import HpmStruct as OldHpmStruct
                from super_gradients.training.utils import HpmStruct

                assert isinstance(OldHpmStruct(a=1), HpmStruct)
        except ImportError:
            self.fail("ImportError raised unexpectedly for HpmStruct")

    def test_deprecated_criterion_params(self):
        with self.assertWarns(DeprecationWarning):
            warnings.simplefilter("always")
            train_params = {
                "max_epochs": 4,
                "lr_decay_factor": 0.1,
                "lr_updates": [4],
                "lr_mode": "StepLRScheduler",
                "lr_warmup_epochs": 0,
                "initial_lr": 0.1,
                "loss": "CrossEntropyLoss",
                "optimizer": "SGD",
                "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "train_metrics_list": [],
                "valid_metrics_list": [],
                "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True,
            }
            train_params = TrainingParams(**train_params)
            train_params.override(criterion_params={"ignore_index": 0})

    def test_train_with_deprecated_criterion_params(self):
        setup_device(device="cpu")
        trainer = Trainer("test_train_with_precise_bn_explicit_size")
        net = ResNet18(num_classes=5, arch_params={})
        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "StepLRScheduler",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "CrossEntropyLoss",
            "criterion_params": {"ignore_index": -300},
            "optimizer": "SGD",
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "precise_bn": True,
            "precise_bn_batch_size": 100,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=10),
            valid_loader=classification_test_dataloader(batch_size=10),
        )

        self.assertEqual(trainer.criterion.ignore_index, -300)


if __name__ == "__main__":
    unittest.main()
