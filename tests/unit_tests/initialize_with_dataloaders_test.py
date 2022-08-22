import unittest

from super_gradients.training import models

from super_gradients import Trainer, ClassificationTestDatasetInterface
import torch
from torch.utils.data import TensorDataset, DataLoader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.exceptions.sg_trainer_exceptions import IllegalDataloaderInitialization


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

    def test_interface_was_not_broken(self):
        trainer = Trainer("test_interface", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)

        net = models.get("efficientnet_b0", arch_params={"num_classes": 5})
        train_params = {"max_epochs": 1, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": torch.nn.CrossEntropyLoss(),
                        "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        trainer.train(net=net, training_params=train_params)

    def test_initialization_rules(self):
        self.assertRaises(IllegalDataloaderInitialization, Trainer, "test_name", model_checkpoints_location='local',
                          train_loader=self.testcase_trainloader)
        self.assertRaises(IllegalDataloaderInitialization, Trainer, "test_name", model_checkpoints_location='local',
                          valid_loader=self.testcase_validloader)
        self.assertRaises(IllegalDataloaderInitialization, Trainer, "test_name", model_checkpoints_location='local',
                          classes=self.testcase_classes)
        self.assertRaises(IllegalDataloaderInitialization, Trainer, "test_name", model_checkpoints_location='local',
                          train_loader=self.testcase_trainloader, valid_loader=self.testcase_validloader)
        self.assertRaises(IllegalDataloaderInitialization, Trainer, "test_name", model_checkpoints_location='local',
                          train_loader=self.testcase_trainloader, classes=self.testcase_classes)
        self.assertRaises(IllegalDataloaderInitialization, Trainer, "test_name", model_checkpoints_location='local',
                          valid_loader=self.testcase_validloader, classes=self.testcase_classes)
        Trainer("test_name", model_checkpoints_location='local', train_loader=self.testcase_trainloader,
                valid_loader=self.testcase_validloader, classes=self.testcase_classes)
        Trainer("test_name", model_checkpoints_location='local', train_loader=self.testcase_trainloader,
                valid_loader=self.testcase_validloader, test_loader=self.testcase_testloader,
                classes=self.testcase_classes)

    def test_train_with_dataloaders(self):
        trainer = Trainer(experiment_name="test_name", model_checkpoints_location="local",
                          train_loader=self.testcase_trainloader, valid_loader=self.testcase_validloader,
                          classes=self.testcase_classes)

        net = models.get("resnet18", arch_params={"num_classes": 5})
        trainer.train(net=net, training_params={"max_epochs": 2,
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
                                       "greater_metric_to_watch_is_better": True})
        self.assertTrue(0 < trainer.best_metric.item() < 1)


if __name__ == '__main__':
    unittest.main()
