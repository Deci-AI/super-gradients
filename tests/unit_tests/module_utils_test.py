from typing import List
import unittest

import torch.nn as nn

from super_gradients.training.models.detection_models.yolov5 import YoLoV5X
from super_gradients.training.utils.module_utils import replace_activations
from super_gradients.training.utils.utils import HpmStruct


class TestModuleUtils(unittest.TestCase):

    def test_activation_replacement(self):
        arch_params = HpmStruct()
        yolov5x = YoLoV5X(arch_params=arch_params)

        new_activation = nn.ReLU()
        activations_to_replace = [nn.SiLU]
        yolov5x_relu = YoLoV5X(arch_params=arch_params)
        replace_activations(yolov5x_relu, new_activation, activations_to_replace)

        self._assert_activations_replaced(yolov5x_relu, yolov5x, new_activation, activations_to_replace)

    def _assert_activations_replaced(self, new_module: nn.Module, orig_module: nn.Module,
                                     new_activation: nn.Module, replaced_activations: List[type]):
        """
        Assert:
            * that new_module doesn't contain any of activations of replaced types
            * that in places where original module has an activation of any of replaced_activations types
            new_module has a new activation
            * that new activations are unique objects and don't share new_activation's address

        Runs recursively on all submodules.

        :param new_module:              A module with replaced activations
        :param orig_module:             A module of the same architecture, but with activations of an original type
        :param new_activation:          A new activation
        :param replaced_activations:    A list of types of activations that should have been replaced;
                                        each should be a subclass of nn.Module
        """
        for new_m, orig_m in zip(new_module.children(), orig_module.children()):
            self.assertTrue(type(new_m) not in replaced_activations)

            if type(orig_m) in replaced_activations:
                self.assertTrue(type(new_m) == type(new_activation))
                self.assertTrue(id(new_m) != id(new_activation))

            self._assert_activations_replaced(new_m, orig_m, new_activation, replaced_activations)


if __name__ == '__main__':
    unittest.main()
