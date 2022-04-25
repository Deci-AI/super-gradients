import unittest
from super_gradients import SgModel, ClassificationTestDatasetInterface
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

    def test_interface_was_not_broken(self):
        model = SgModel("test_interface", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)

        model.build_model("efficientnet_b0")
        train_params = {"max_epochs": 1, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": torch.nn.CrossEntropyLoss(),
                        "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        model.train(train_params)

    def test_initialization_rules(self):
        self.assertRaises(ValueError, SgModel, "test_name", model_checkpoints_location='local',
                          train_loader=self.testcase_trainloader)
        self.assertRaises(ValueError, SgModel, "test_name", model_checkpoints_location='local',
                          valid_loader=self.testcase_validloader)
        self.assertRaises(ValueError, SgModel, "test_name", model_checkpoints_location='local',
                          classes=self.testcase_classes)
        self.assertRaises(ValueError, SgModel, "test_name", model_checkpoints_location='local',
                          train_loader=self.testcase_trainloader, valid_loader=self.testcase_validloader)
        self.assertRaises(ValueError, SgModel, "test_name", model_checkpoints_location='local',
                          train_loader=self.testcase_trainloader, classes=self.testcase_classes)
        self.assertRaises(ValueError, SgModel, "test_name", model_checkpoints_location='local',
                          valid_loader=self.testcase_validloader, classes=self.testcase_classes)
        SgModel("test_name", model_checkpoints_location='local', train_loader=self.testcase_trainloader,
                valid_loader=self.testcase_validloader, classes=self.testcase_classes)
        SgModel("test_name", model_checkpoints_location='local', train_loader=self.testcase_trainloader,
                valid_loader=self.testcase_validloader, test_loader=self.testcase_testloader,
                classes=self.testcase_classes)

    def test_train_with_dataloaders(self):
        model = SgModel(experiment_name="test_name", model_checkpoints_location="local",
                        train_loader=self.testcase_trainloader, valid_loader=self.testcase_validloader,
                        classes=self.testcase_classes)

        model.build_model("resnet18")
        model.train(training_params={"max_epochs": 2,
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
        self.assertTrue(0 < model.best_metric.item() < 1)


if __name__ == '__main__':
    unittest.main()
