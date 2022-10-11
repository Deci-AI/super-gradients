import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.training.models.all_architectures import ARCHITECTURES
from super_gradients.training.models.model_registry import register_model


class ModelRegistryTest(unittest.TestCase):

    def setUp(self):
        @register_model('myconvnet')
        class MyConvNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, num_classes)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        @register_model()
        def myconvnet_for_cifar10():
            return MyConvNet(num_classes=10)

    def tearDown(self):
        ARCHITECTURES.pop('myconvnet', None)
        ARCHITECTURES.pop('myconvnet_for_cifar10', None)

    def test_cls_is_registered(self):
        assert ARCHITECTURES['myconvnet']

    def test_fn_is_registered(self):
        assert ARCHITECTURES['myconvnet_for_cifar10']

    def test_model_is_instantiable(self):
        assert ARCHITECTURES['myconvnet_for_cifar10']()
        assert ARCHITECTURES['myconvnet'](num_classes=10)

    def test_model_outputs(self):
        torch.manual_seed(0)
        model_1 = ARCHITECTURES['myconvnet_for_cifar10']()
        torch.manual_seed(0)
        model_2 = ARCHITECTURES['myconvnet'](num_classes=10)
        dummy_input = torch.randn(1, 3, 32, 32, requires_grad=False)
        x = model_1(dummy_input)
        y = model_2(dummy_input)
        assert torch.equal(x, y)

    def test_existing_key(self):
        with self.assertRaises(Exception):
            @register_model()
            def myconvnet_for_cifar10():
                return


if __name__ == '__main__':
    unittest.main()
