import unittest

from torch import Tensor
from torchmetrics import Accuracy
import torch
from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader


class CriterionWithUnnamedComponents(torch.nn.CrossEntropyLoss):
    def __init__(self):
        super(CriterionWithUnnamedComponents, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> tuple:
        loss = super(CriterionWithUnnamedComponents, self).forward(input=input, target=target)
        items = torch.cat((loss.unsqueeze(0), loss.unsqueeze(0))).detach()
        return loss, items


class CriterionWithNamedComponents(CriterionWithUnnamedComponents):
    def __init__(self):
        super(CriterionWithNamedComponents, self).__init__()
        self.component_names = ["loss_A", "loss_B"]


class LossLoggingsTest(unittest.TestCase):
    def test_single_item_logging(self):
        trainer = Trainer("test_single_item_logging", model_checkpoints_location="local")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        train_params = {
            "max_epochs": 1,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": torch.nn.CrossEntropyLoss(),
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)
        self.assertListEqual(trainer.loss_logging_items_names, ["CrossEntropyLoss"])

    def test_multiple_unnamed_components_loss_logging(self):
        trainer = Trainer("test_multiple_unnamed_components_loss_logging", model_checkpoints_location="local")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        train_params = {
            "max_epochs": 1,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": CriterionWithUnnamedComponents(),
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)
        self.assertListEqual(trainer.loss_logging_items_names, ["CriterionWithUnnamedComponents/loss_0", "CriterionWithUnnamedComponents/loss_1"])

    def test_multiple_named_components_loss_logging(self):
        trainer = Trainer("test_multiple_named_components_loss_logging", model_checkpoints_location="local")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        train_params = {
            "max_epochs": 1,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": CriterionWithNamedComponents(),
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)
        self.assertListEqual(trainer.loss_logging_items_names, ["CriterionWithNamedComponents/loss_A", "CriterionWithNamedComponents/loss_B"])


if __name__ == "__main__":
    unittest.main()
