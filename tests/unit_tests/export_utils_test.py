import copy

import torch
from torch import nn
import unittest

from super_gradients.common.registry.registry import ARCHITECTURES
from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.utils.export_utils import fuse_conv_bn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestUtil(unittest.TestCase):
    def test_fuse_conv_bn_real_archs(self):
        """
        test the fuse_conv_bn function. run the function on some Sg architectures and assert
        the result of the original net are the same as the results of the fused net
        """

        archs = ["resnet18", "mobilenet_v2", "densenet121", "regnetY200"]

        for arch_name in archs:

            model1 = ARCHITECTURES[arch_name](HpmStruct(**{"num_classes": 10, "dropout": 0.1}))
            model2 = copy.deepcopy(model1)

            model1.eval()
            model2.eval()

            fuse_conv_bn(model2, True)

            input = torch.rand(size=(1, 3, 320, 320))
            output1 = model1(input)[0]
            output2 = model2(input)[0]

            param_count1 = count_parameters(model1)
            param_count2 = count_parameters(model2)

            self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
            print(f"Tested fuse Conv BN on {arch_name}: OK ({param_count1 - param_count2} less params)")

    def test_fuse_conv_bn_on_sequential_models(self):

        # assert the bn module was replaced with Identity
        model = nn.Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3))
        model.eval()
        fuse_conv_bn(model, replace_bn_with_identity=True)
        self.assertEqual(len(model._modules), 2)
        self.assertIsInstance(model._modules["0"], nn.Conv2d)
        self.assertIsInstance(model._modules["1"], nn.Identity)

        # assert the bn module was removed
        model = nn.Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3))
        model.eval()
        fuse_conv_bn(model, replace_bn_with_identity=False)
        self.assertEqual(len(model._modules), 1)
        self.assertIsInstance(model._modules["0"], nn.Conv2d)

        # assert all bn module were removed
        model = nn.Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3))
        model.eval()
        fuse_conv_bn(model, replace_bn_with_identity=False)
        self.assertEqual(len(model._modules), 2)
        self.assertIsInstance(model._modules["0"], nn.Conv2d)

        # assert only merged bn module were removed
        model = nn.Sequential(nn.Conv2d(3, 3, 3), nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3))
        model.eval()
        fuse_conv_bn(model, replace_bn_with_identity=False)
        self.assertEqual(len(model._modules), 2)
        self.assertIsInstance(model._modules["0"], nn.Conv2d)
        self.assertIsInstance(model._modules["1"], nn.Conv2d)

    def test_fuse_conv_bn_on_toy_models(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, 3)
                self.bn1 = nn.BatchNorm2d(3)
                self.conv2 = nn.Conv2d(3, 3, 3)
                self.bn2 = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                return x

        # assert the bn module was replaced with Identity
        model = Toy()
        model.eval()
        fuse_conv_bn(model, replace_bn_with_identity=True)
        self.assertIsNotNone(model.bn1)
        self.assertIsInstance(model.conv1, nn.Conv2d)
        self.assertIsInstance(model.bn1, nn.Identity)

        # assert the bn module was removed
        model = Toy()
        model.eval()
        fuse_conv_bn(model, replace_bn_with_identity=False)
        self.assertFalse(hasattr(model, "bn1"))
        self.assertIsInstance(model.conv1, nn.Conv2d)

        # assert all bn module were removed
        model = Toy()
        model.eval()
        fuse_conv_bn(model, replace_bn_with_identity=False)
        self.assertFalse(hasattr(model, "bn1"))
        self.assertIsInstance(model.conv1, nn.Conv2d)
        self.assertFalse(hasattr(model, "bn2"))
        self.assertIsInstance(model.conv2, nn.Conv2d)

        # assert correct number of parameters removed
        model = Toy()
        model.eval()
        before = count_parameters(model)
        fuse_conv_bn(model, replace_bn_with_identity=False)
        after = count_parameters(model)
        self.assertEqual(before - after, 12)  # each bn of 3 channels has 6 parameters (12 together)


if __name__ == "__main__":
    unittest.main()
