import unittest
from typing import Union

from omegaconf import DictConfig
from torch import nn

from super_gradients.common.registry import register_model
from super_gradients.training import models
from super_gradients.training.models import CustomizableDetector, get_arch_params
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


if __name__ == "__main__":
    unittest.main()
