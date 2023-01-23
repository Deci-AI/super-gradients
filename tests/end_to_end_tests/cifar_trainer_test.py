import unittest

from super_gradients.common.object_names import Models
from super_gradients.training import models

import super_gradients

from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    cifar10_train,
    cifar10_val,
    cifar100_train,
    cifar100_val,
)


class TestCifarTrainer(unittest.TestCase):
    def test_train_cifar10_dataloader(self):
        super_gradients.init_trainer()
        trainer = Trainer("test")
        cifar10_train_dl, cifar10_val_dl = cifar10_train(), cifar10_val()
        model = models.get(Models.RESNET18_CIFAR, arch_params={"num_classes": 10})
        trainer.train(
            model=model,
            training_params={
                "max_epochs": 1,
                "initial_lr": 0.1,
                "loss": "cross_entropy",
                "train_metrics_list": ["Accuracy"],
                "valid_metrics_list": ["Accuracy"],
                "metric_to_watch": "Accuracy",
            },
            train_loader=cifar10_train_dl,
            valid_loader=cifar10_val_dl,
        )

    def test_train_cifar100_dataloader(self):
        super_gradients.init_trainer()
        trainer = Trainer("test")
        cifar100_train_dl, cifar100_val_dl = cifar100_train(), cifar100_val()
        model = models.get(Models.RESNET18_CIFAR, arch_params={"num_classes": 100})
        trainer.train(
            model=model,
            training_params={
                "max_epochs": 1,
                "initial_lr": 0.1,
                "loss": "cross_entropy",
                "train_metrics_list": ["Accuracy"],
                "valid_metrics_list": ["Accuracy"],
                "metric_to_watch": "Accuracy",
            },
            train_loader=cifar100_train_dl,
            valid_loader=cifar100_val_dl,
        )


if __name__ == "__main__":
    unittest.main()
