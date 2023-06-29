import unittest
import torch
from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from torchmetrics import Metric
from super_gradients.training.utils.callbacks import Phase


class DummyMetric(Metric):
    def update(self, *args, **kwargs) -> None:
        pass

    def compute(self):
        return 1


class TrainWithTorchSchedulerTest(unittest.TestCase):
    def test_train_with_StepLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_StepLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 2,
            "lr_mode": {"StepLR": {"gamma": 0.1, "step_size": 1, "phase": Phase.TRAIN_EPOCH_END}},
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
        self.assertAlmostEqual(0.001, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_LambdaLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_LambdaLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        def lr_compute_fn(epoch):
            return 1 / (epoch + 10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 2,
            "lr_mode": {"LambdaLR": {"lr_lambda": lr_compute_fn, "phase": Phase.TRAIN_EPOCH_END}},
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
        self.assertAlmostEqual(0.1 / 12, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_MultiStepLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_MultiStepLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 2,
            "lr_mode": {"MultiStepLR": {"milestones": [0, 1], "phase": Phase.TRAIN_EPOCH_END}},
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
        self.assertAlmostEqual(0.001, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_ConstantLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_ConstantLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 5,
            "lr_mode": {"ConstantLR": {"factor": 0.5, "total_iters": 4, "phase": Phase.TRAIN_EPOCH_END}},
            "lr_warmup_epochs": 0,
            "initial_lr": 0.05,
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
        self.assertAlmostEqual(0.05, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_CosineAnnealingLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_CosineAnnealingLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 3,
            "lr_mode": {"CosineAnnealingLR": {"T_max": 3, "phase": Phase.TRAIN_EPOCH_END}},
            "lr_warmup_epochs": 0,
            "initial_lr": 0.05,
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
        self.assertAlmostEqual(0.0, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_CosineAnnealingWarmRestarts_torch_scheduler(self):
        trainer = Trainer("test_train_with_CosineAnnealingWarmRestarts_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 4,
            "lr_mode": {"CosineAnnealingWarmRestarts": {"T_0": 2, "phase": Phase.TRAIN_EPOCH_END}},
            "lr_warmup_epochs": 0,
            "initial_lr": 0.05,
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
        self.assertAlmostEqual(0.05, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_CyclicLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_CyclicLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 4,
            "lr_mode": {"CyclicLR": {"base_lr": 0.01, "max_lr": 0.1, "phase": Phase.TRAIN_EPOCH_END}},
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
        self.assertAlmostEqual(0.01018, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_ExponentialLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_ExponentialLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 4,
            "lr_mode": {"ExponentialLR": {"gamma": 0.01, "phase": Phase.TRAIN_EPOCH_END}},
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
        self.assertAlmostEqual(1e-9, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_LinearLR_torch_scheduler(self):
        trainer = Trainer("test_train_with_LinearLR_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 4,
            "lr_mode": {"LinearLR": {"phase": Phase.TRAIN_EPOCH_END}},
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
        self.assertAlmostEqual(0.08666666666666668, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_ReduceLROnPlateau_torch_scheduler(self):
        trainer = Trainer("test_train_with_ReduceLROnPlateau_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 2,
            "lr_decay_factor": 0.1,
            "lr_mode": {"ReduceLROnPlateau": {"patience": 0, "phase": Phase.TRAIN_EPOCH_END, "metric_name": "DummyMetric"}},
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": torch.nn.CrossEntropyLoss(),
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [DummyMetric()],
            "valid_metrics_list": [DummyMetric()],
            "metric_to_watch": "DummyMetric",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)
        self.assertAlmostEqual(0.01, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_resume_train_with_torch_scheduler(self):
        trainer = Trainer("test_resume_train_with_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 2,
            "lr_mode": {"StepLR": {"gamma": 0.1, "step_size": 1, "phase": Phase.TRAIN_EPOCH_END}},
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

        train_params["max_epochs"] = 3
        train_params["resume"] = True

        trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)

        self.assertAlmostEqual(0.0001, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_resume_train_with_ReduceLROnPlateau_torch_scheduler(self):
        trainer = Trainer("test_resume_train_with_ReduceLROnPlateau_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": 2,
            "lr_decay_factor": 0.1,
            "lr_mode": {"ReduceLROnPlateau": {"patience": 0, "phase": Phase.TRAIN_EPOCH_END, "metric_name": "DummyMetric"}},
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": torch.nn.CrossEntropyLoss(),
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [DummyMetric()],
            "valid_metrics_list": [DummyMetric()],
            "metric_to_watch": "DummyMetric",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)

        train_params["max_epochs"] = 3
        train_params["resume"] = True

        trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)

        self.assertAlmostEqual(0.001, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)


if __name__ == "__main__":
    unittest.main()
