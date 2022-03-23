import unittest
from super_gradients import SgModel, \
    ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy, Top5, ToyTestClassificationMetric
from super_gradients.training.models import ResNet18
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from super_gradients.training.utils.callbacks import LRSchedulerCallback, Phase
from torchmetrics import F1
import torch
import numpy as np
from super_gradients.training.datasets.dataset_interfaces import DatasetInterface


class TrainWithInitializedObjectsTest(unittest.TestCase):
    """
    Unit test for training with initialized objects passed as parameters.
    """
    def test_train_with_external_criterion(self):
        model = SgModel("external_criterion_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)

        net = ResNet18(num_classes=5, arch_params={})
        model.build_model(net, load_checkpoint=False)
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": torch.nn.CrossEntropyLoss(), "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        model.train(train_params)

    def test_train_with_external_optimizer(self):
        model = SgModel("external_optimizer_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)

        net = ResNet18(num_classes=5, arch_params={})
        optimizer = SGD(params=net.parameters(), lr=0.1)
        model.build_model(net, load_checkpoint=False)
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": optimizer,
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        model.train(train_params)

    def test_train_with_external_scheduler(self):
        model = SgModel("external_scheduler_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)

        lr = 0.3
        net = ResNet18(num_classes=5, arch_params={})
        optimizer = SGD(params=net.parameters(), lr=lr)
        lr_scheduler = MultiStepLR(optimizer=optimizer, milestones=[1, 2], gamma=0.1)
        phase_callbacks = [LRSchedulerCallback(lr_scheduler, Phase.TRAIN_EPOCH_END)]
        model.build_model(net, load_checkpoint=False)

        train_params = {"max_epochs": 2, "phase_callbacks": phase_callbacks,
                        "lr_warmup_epochs": 0, "initial_lr": lr, "loss": "cross_entropy", "optimizer": optimizer,
                        "criterion_params": {},
                        "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        model.train(train_params)
        assert lr_scheduler.get_last_lr()[0] == lr * 0.1 * 0.1

    def test_train_with_reduce_on_plateau(self):
        model = SgModel("external_reduce_on_plateau_scheduler_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)

        lr = 0.3
        net = ResNet18(num_classes=5, arch_params={})
        optimizer = SGD(params=net.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=0)
        phase_callbacks = [LRSchedulerCallback(lr_scheduler, Phase.VALIDATION_EPOCH_END, "ToyTestClassificationMetric")]
        model.build_model(net, load_checkpoint=False)

        train_params = {"max_epochs": 2, "phase_callbacks": phase_callbacks,
                        "lr_warmup_epochs": 0, "initial_lr": lr, "loss": "cross_entropy", "optimizer": optimizer,
                        "criterion_params": {},
                        "train_metrics_list": [Accuracy(), Top5()],
                        "valid_metrics_list": [Accuracy(), Top5(), ToyTestClassificationMetric()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        model.train(train_params)
        assert lr_scheduler._last_lr[0] == lr * 0.1

    def test_train_with_external_metric(self):
        model = SgModel("external_metric_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)

        net = ResNet18(num_classes=5, arch_params={})
        model.build_model(net, load_checkpoint=False)
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [F1()], "valid_metrics_list": [F1()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "F1",
                        "greater_metric_to_watch_is_better": True}
        model.train(train_params)

    def test_train_with_external_dataloaders(self):
        model = SgModel("external_data_loader_test", model_checkpoints_location='local')

        batch_size = 5
        trainset = torch.utils.data.TensorDataset(torch.Tensor(np.random.random((10, 3, 32, 32))), torch.LongTensor(np.zeros((10))))

        valset = torch.utils.data.TensorDataset(torch.Tensor(np.random.random((10, 3, 32, 32))), torch.LongTensor(np.zeros((10))))

        classes = [0, 1, 2, 3, 4]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

        dataset_interface = DatasetInterface(train_loader=train_loader, val_loader=val_loader, classes=classes)
        model.connect_dataset_interface(dataset_interface)

        net = ResNet18(num_classes=5, arch_params={})
        model.build_model(net, load_checkpoint=False)
        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [F1()], "valid_metrics_list": [F1()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "F1",
                        "greater_metric_to_watch_is_better": True}
        model.train(train_params)


if __name__ == '__main__':
    unittest.main()
