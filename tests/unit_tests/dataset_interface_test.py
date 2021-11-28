import unittest
from super_gradients.training.datasets import Cifar10DatasetInterface


class TestDatasetInterface(unittest.TestCase):
    def test_cifar(self):
        test_dataset_interface = Cifar10DatasetInterface()
        cifar_dataset_sample = test_dataset_interface.get_test_sample()
        self.assertListEqual([3, 32, 32], list(cifar_dataset_sample[0].shape))


if __name__ == '__main__':
    unittest.main()
