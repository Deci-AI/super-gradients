import unittest

from super_gradients.common.object_names import Models
from super_gradients.training import models

from super_gradients import Trainer
import torch
from torch.utils.data import TensorDataset, DataLoader
from super_gradients.training.metrics import Accuracy


class InitializeWithDataloadersTest(unittest.TestCase):
    def setUp(self):
        self.testcase_classes = [0, 1, 2, 3, 4]
        train_size, valid_size, test_size = 160, 20, 20
        channels, width, height = 3, 224, 224
        inp = torch.randn((train_size, channels, width, height))
        label = torch.randint(0, len(self.testcase_classes), size=(train_size,))
        self.testcase_trainloader = DataLoader(TensorDataset(inp, label))

        inp = torch.randn((valid_size, channels, width, height))
        label = torch.randint(0, len(self.testcase_classes), size=(valid_size,))
        self.testcase_validloader = DataLoader(TensorDataset(inp, label))

        inp = torch.randn((test_size, channels, width, height))
        label = torch.randint(0, len(self.testcase_classes), size=(test_size,))
        self.testcase_testloader = DataLoader(TensorDataset(inp, label))

    def test_train_with_dataloaders(self):
        trainer = Trainer(experiment_name="test_name")
        model = models.get(Models.RESNET18, num_classes=5)
        trainer.train(
            model=model,
            training_params={
                "max_epochs": 2,
                "lr_updates": [5, 6, 12],
                "lr_decay_factor": 0.01,
                "lr_mode": "step",
                "initial_lr": 0.01,
                "loss": "cross_entropy",
                "optimizer": "SGD",
                "optimizer_params": {"weight_decay": 1e-5, "momentum": 0.9},
                "train_metrics_list": [Accuracy()],
                "valid_metrics_list": [Accuracy()],
                "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True,
            },
            train_loader=self.testcase_trainloader,
            valid_loader=self.testcase_validloader,
        )
        self.assertTrue(0 < trainer.best_metric.item() < 1)


if __name__ == "__main__":
    unittest.main()
