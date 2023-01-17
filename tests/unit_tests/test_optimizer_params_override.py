import unittest

from super_gradients.training.utils.utils import get_param

from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18


class TrainOptimizerParamsOverride(unittest.TestCase):
    def test_optimizer_params_partial_override(self):
        trainer = Trainer("test_optimizer_params_partial_override")
        net = ResNet18(num_classes=5, arch_params={})
        train_params = {
            "max_epochs": 1,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"momentum": 0.9},
            "zero_weight_decay_on_bias_and_bn": True,
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=10),
            valid_loader=classification_test_dataloader(batch_size=10),
        )
        self.assertTrue(get_param(trainer.training_params.optimizer_params, "weight_decay"), 1e-4)
        self.assertTrue(get_param(trainer.training_params.optimizer_params, "momentum"), 0.9)

    def test_optimizer_params_full_override(self):
        trainer = Trainer("test_optimizer_params_full_override")
        net = ResNet18(num_classes=5, arch_params={})
        train_params = {
            "max_epochs": 1,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "zero_weight_decay_on_bias_and_bn": True,
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=10),
            valid_loader=classification_test_dataloader(batch_size=10),
        )
        self.assertTrue(get_param(trainer.training_params.optimizer_params, "weight_decay"), 1e-4)
        self.assertTrue(get_param(trainer.training_params.optimizer_params, "momentum"), 0.9)
