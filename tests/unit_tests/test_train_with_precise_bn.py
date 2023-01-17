import unittest

from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18


class TrainWithPreciseBNTest(unittest.TestCase):
    """
    Unit test for training with precise_bn.
    """

    def test_train_with_precise_bn_explicit_size(self):
        trainer = Trainer("test_train_with_precise_bn_explicit_size")
        net = ResNet18(num_classes=5, arch_params={})
        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
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

    def test_train_with_precise_bn_implicit_size(self):
        trainer = Trainer("test_train_with_precise_bn_implicit_size")

        net = ResNet18(num_classes=5, arch_params={})
        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "precise_bn": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=10),
            valid_loader=classification_test_dataloader(batch_size=10),
        )


if __name__ == "__main__":
    unittest.main()
