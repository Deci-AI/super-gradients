import unittest

from super_gradients.training import models

import super_gradients

from super_gradients import SgModel
from super_gradients.training.datasets.dataset_interfaces import LibraryDatasetInterface


class TestCifar10Trainer(unittest.TestCase):
    def test_train_cifar10(self):
        super_gradients.init_trainer()
        model = SgModel("test", model_checkpoints_location='local')
        cifar_10_dataset_interface = LibraryDatasetInterface(name="cifar10")
        model.connect_dataset_interface(cifar_10_dataset_interface)
        net = models.get("resnet18_cifar", arch_params={"num_classes": 10})
        model.train(net=net, training_params={"max_epochs": 1})


if __name__ == '__main__':
    unittest.main()
