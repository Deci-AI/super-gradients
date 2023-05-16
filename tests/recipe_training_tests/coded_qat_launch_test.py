import unittest

from torchvision.transforms import Normalize, ToTensor, RandomHorizontalFlip, RandomCrop

from super_gradients import Trainer
from super_gradients.training import modify_params_for_qat
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18


class CodedQATLuanchTest(unittest.TestCase):
    def test_qat_launch(self):
        trainer = Trainer("test_launch_qat_with_minimal_changes")
        net = ResNet18(num_classes=10, arch_params={})
        train_params = {
            "max_epochs": 10,
            "lr_updates": [],
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
            "ema": True,
        }

        train_dataset_params = {
            "transforms": [
                RandomCrop(size=32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        }

        train_dataloader_params = {"batch_size": 256}

        val_dataset_params = {"transforms": [ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]}

        val_dataloader_params = {"batch_size": 256}

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params = modify_params_for_qat(
            train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
        )

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.qat(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
            calib_loader=train_loader,
        )

    def test_ptq_launch(self):
        trainer = Trainer("test_launch_ptq_with_minimal_changes")
        net = ResNet18(num_classes=10, arch_params={})
        train_params = {
            "max_epochs": 10,
            "lr_updates": [],
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
            "ema": True,
        }

        train_dataset_params = {
            "transforms": [
                RandomCrop(size=32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        }

        train_dataloader_params = {"batch_size": 256}

        val_dataset_params = {"transforms": [ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])]}

        val_dataloader_params = {"batch_size": 256}

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params = modify_params_for_qat(
            train_params, train_dataset_params, val_dataset_params, train_dataloader_params, val_dataloader_params
        )

        train_loader = cifar10_train(dataset_params=train_dataset_params, dataloader_params=train_dataloader_params)
        valid_loader = cifar10_val(dataset_params=val_dataset_params, dataloader_params=val_dataloader_params)

        trainer.ptq(model=net, valid_loader=valid_loader, calib_loader=train_loader, valid_metrics_list=train_params["valid_metrics_list"])


if __name__ == "__main__":
    unittest.main()
