import unittest

import torch

from super_gradients import Trainer
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5


class FactoriesTest(unittest.TestCase):

    def test_training_with_factories(self):
        trainer = Trainer("test_train_with_factories", model_checkpoints_location='local')
        net = models.get("resnet18", num_classes=5)
        train_params = {"max_epochs": 2,
                        "lr_updates": [1],
                        "lr_decay_factor": 0.1,
                        "lr_mode": "step",
                        "lr_warmup_epochs": 0,
                        "initial_lr": 0.1,
                        "loss": "cross_entropy",
                        "optimizer": "torch.optim.ASGD",    # use an optimizer by factory
                        "criterion_params": {},
                        "optimizer_params": {"lambd": 0.0001, "alpha": 0.75},
                        "train_metrics_list": ["Accuracy", "Top5"],  # use a metric by factory
                        "valid_metrics_list": ["Accuracy", "Top5"],  # use a metric by factory
                         "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}

        trainer.train(model=net, training_params=train_params,
                      train_loader=classification_test_dataloader(),
                      valid_loader=classification_test_dataloader())

        self.assertIsInstance(trainer.train_metrics.Accuracy, Accuracy)
        self.assertIsInstance(trainer.valid_metrics.Top5, Top5)
        self.assertIsInstance(trainer.optimizer, torch.optim.ASGD)


if __name__ == '__main__':
    unittest.main()
