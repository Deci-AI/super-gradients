import unittest
from super_gradients.common.registry.registry import ARCHITECTURES
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils.utils import HpmStruct
import torch


class AllArchitecturesTest(unittest.TestCase):
    def setUp(self):
        # contains all arch_params needed for initialization of all architectures
        self.all_arch_params = HpmStruct(
            **{
                "num_classes": 10,
                "width_mult": 1,
                "threshold": 1,
                "sml_net": torch.nn.Identity(),
                "big_net": torch.nn.Identity(),
                "dropout": 0,
                "build_residual_branches": True,
            }
        )

    def test_architecture_is_sg_module(self):
        """
        Validate all models from all_architectures.py are SgModule
        """
        for arch_name in ARCHITECTURES:
            # skip custom constructors to keep all_arch_params as general as a possible
            if "custom" in arch_name.lower() or "nas" in arch_name.lower() or "kd" in arch_name.lower():
                continue
            self.assertTrue(isinstance(ARCHITECTURES[arch_name](arch_params=self.all_arch_params), SgModule))


if __name__ == "__main__":
    unittest.main()
