import unittest

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchmetrics import F1Score

from super_gradients import Trainer, \
    ClassificationTestDatasetInterface
from super_gradients.training import models
from super_gradients.training.datasets.dataset_interfaces import DatasetInterface
from super_gradients.training.metrics import Accuracy, Top5, ToyTestClassificationMetric
from super_gradients.training.utils.callbacks import LRSchedulerCallback, Phase


class TrainWithInitializedObjectsTest(unittest.TestCase):
    """
    Unit test for training with initialized objects passed as parameters.
    """

    def test_train_with_external_criterion(self):
        trainer = Trainer("external_criterion_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)

        model = models.get("resnet18", arch_params={"num_classes": 5})
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": torch.nn.CrossEntropyLoss(),
                        "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(model=model, training_params=train_params)

    def test_train_with_external_optimizer(self):
        trainer = Trainer("external_optimizer_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)

        model = models.get("resnet18", arch_params={"num_classes": 5})
        optimizer = SGD(params=model.parameters(), lr=0.1)
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": optimizer,
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(model=model, training_params=train_params)

    def test_train_with_external_scheduler(self):
        trainer = Trainer("external_scheduler_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)

        lr = 0.3
        model = models.get("resnet18", arch_params={"num_classes": 5})
        optimizer = SGD(params=model.parameters(), lr=lr)
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[1, 2], gamma=0.1)
        phase_callbacks = [LRSchedulerCallback(lr_scheduler, Phase.TRAIN_EPOCH_END)]

        train_params = {"max_epochs": 2, "phase_callbacks": phase_callbacks,
                        "lr_warmup_epochs": 0, "initial_lr": lr, "loss": "cross_entropy", "optimizer": optimizer,
                        "criterion_params": {},
                        "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(model=model, training_params=train_params)
        assert lr_scheduler.get_last_lr()[0] == lr * 0.1 * 0.1

    def test_train_with_external_scheduler_class(self):
        trainer = Trainer("external_scheduler_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)

        model = models.get("resnet18", arch_params={"num_classes": 5})
        optimizer = SGD  # a class - not an instance

        train_params = {"max_epochs": 2,
                        "lr_warmup_epochs": 0, "initial_lr": 0.3, "loss": "cross_entropy", "optimizer": optimizer,
                        "criterion_params": {},
                        "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(model=model, training_params=train_params)

    def test_train_with_reduce_on_plateau(self):
        trainer = Trainer("external_reduce_on_plateau_scheduler_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)

        lr = 0.3
        model = models.get("resnet18", arch_params={"num_classes": 5})
        optimizer = SGD(params=model.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=0)
        phase_callbacks = [LRSchedulerCallback(lr_scheduler, Phase.VALIDATION_EPOCH_END, "ToyTestClassificationMetric")]

        train_params = {"max_epochs": 2, "phase_callbacks": phase_callbacks,
                        "lr_warmup_epochs": 0, "initial_lr": lr, "loss": "cross_entropy", "optimizer": optimizer,
                        "criterion_params": {},
                        "train_metrics_list": [Accuracy(), Top5()],
                        "valid_metrics_list": [Accuracy(), Top5(), ToyTestClassificationMetric()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(model=model, training_params=train_params)
        assert lr_scheduler._last_lr[0] == lr * 0.1

    def test_train_with_external_metric(self):
        trainer = Trainer("external_metric_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)

        model = models.get("resnet18", arch_params={"num_classes": 5})
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [F1Score()], "valid_metrics_list": [F1Score()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "F1Score",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(model=model, training_params=train_params)

    def test_train_with_external_dataloaders(self):
        trainer = Trainer("external_data_loader_test", model_checkpoints_location='local')

        batch_size = 5
        trainset = torch.utils.data.TensorDataset(torch.Tensor(np.random.random((10, 3, 32, 32))),
                                                  torch.LongTensor(np.zeros((10))))

        valset = torch.utils.data.TensorDataset(torch.Tensor(np.random.random((10, 3, 32, 32))),
                                                torch.LongTensor(np.zeros((10))))

        classes = [0, 1, 2, 3, 4]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

        dataset_interface = DatasetInterface(train_loader=train_loader, val_loader=val_loader, classes=classes)
        trainer.connect_dataset_interface(dataset_interface)

        model = models.get("resnet18", arch_params={"num_classes": 5})
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [F1Score()], "valid_metrics_list": [F1Score()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "F1Score",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(model=model, training_params=train_params)


if __name__ == '__main__':
    unittest.main()
