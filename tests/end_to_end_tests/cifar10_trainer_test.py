import unittest

import super_gradients

from super_gradients import Trainer
from super_gradients.training.datasets.dataset_interfaces import LibraryDatasetInterface


class TestCifar10Trainer(unittest.TestCase):
    def test_train_cifar10(self):
        super_gradients.init_trainer()
        trainer = Trainer("test", model_checkpoints_location='local')
        cifar_10_dataset_interface = LibraryDatasetInterface(name="cifar10")
        trainer.connect_dataset_interface(cifar_10_dataset_interface)
        trainer.build_model("resnet18_cifar", arch_params={'num_classes': 10})
        trainer.train(training_params={"max_epochs": 1})


if __name__ == '__main__':
    unittest.main()
