import unittest
import torch
from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from torchmetrics import Metric
from super_gradients.training.utils.callbacks import Phase


class DummyMetric(Metric):
    def update(self, *args, **kwargs) -> None:
        pass

    def compute(self):
        return 1


class TrainWithTorchSchedulerTest(unittest.TestCase):
    def _run_scheduler_test(self, scheduler_name, scheduler_params, expected_lr, epochs=2, test_resume=False):
        trainer = Trainer("test_" + scheduler_name + "_torch_scheduler")
        dataloader = classification_test_dataloader(batch_size=10)

        model = models.get(Models.RESNET18, num_classes=5)
        train_params = {
            "max_epochs": epochs,
            "lr_mode": {scheduler_name: scheduler_params},
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
        if test_resume:
            train_params["max_epochs"] = epochs + 1
            train_params["resume"] = True
            trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)

        self.assertAlmostEqual(expected_lr, trainer.optimizer.param_groups[0]["lr"], delta=1e-8)

    def test_train_with_StepLR_torch_scheduler(self):
        scheduler_params = {"gamma": 0.1, "step_size": 1, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("StepLR", scheduler_params, 0.001)

    def test_train_with_LambdaLR_torch_scheduler(self):
        def lr_compute_fn(epoch):
            return 1 / (epoch + 10)

        scheduler_params = {"lr_lambda": lr_compute_fn, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("LambdaLR", scheduler_params, 0.1 / 12)

    def test_train_with_MultiStepLR_torch_scheduler(self):
        scheduler_params = {"milestones": [0, 1], "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("MultiStepLR", scheduler_params, 0.001)

    def test_train_with_ConstantLR_torch_scheduler(self):
        scheduler_params = {"factor": 0.5, "total_iters": 4, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("ConstantLR", scheduler_params, 0.05)

    def test_train_with_CosineAnnealingLR_torch_scheduler(self):
        scheduler_params = {"T_max": 3, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("CosineAnnealingLR", scheduler_params, 0.025)

    def test_train_with_CosineAnnealingWarmRestarts_torch_scheduler(self):
        scheduler_params = {"T_0": 2, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("CosineAnnealingWarmRestarts", scheduler_params, 0.1, 4)

    def test_train_with_CyclicLR_torch_scheduler(self):
        scheduler_params = {"base_lr": 0.01, "max_lr": 0.1, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("CyclicLR", scheduler_params, 0.01018, 4)

    def test_train_with_ExponentialLR_torch_scheduler(self):
        scheduler_params = {"gamma": 0.01, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("ExponentialLR", scheduler_params, 1e-09, 4)

    def test_train_with_LinearLR_torch_scheduler(self):
        scheduler_params = {"phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("LinearLR", scheduler_params, 0.08666666666666668, 4)

    def test_train_with_ReduceLROnPlateau_torch_scheduler(self):
        scheduler_params = {"patience": 0, "phase": Phase.TRAIN_EPOCH_END, "metric_name": "DummyMetric"}
        self._run_scheduler_test("ReduceLROnPlateau", scheduler_params, 0.01)

    def test_resume_train_with_torch_scheduler(self):
        scheduler_params = {"gamma": 0.1, "step_size": 1, "phase": Phase.TRAIN_EPOCH_END}
        self._run_scheduler_test("StepLR", scheduler_params, 0.0001, 2, True)

    def test_resume_train_with_ReduceLROnPlateau_torch_scheduler(self):
        scheduler_params = {"patience": 0, "phase": Phase.TRAIN_EPOCH_END, "metric_name": "DummyMetric"}
        self._run_scheduler_test("ReduceLROnPlateau", scheduler_params, 0.001, 2, True)


if __name__ == "__main__":
    unittest.main()
