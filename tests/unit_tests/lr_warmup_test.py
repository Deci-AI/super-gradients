import unittest

import numpy as np

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models import LeNet
from super_gradients.training.utils.callbacks import TestLRCallback, LRCallbackBase, Phase, Callback, PhaseContext, CosineLRCallback


class CollectLRCallback(Callback):
    def __init__(self):
        self.per_step_learning_rates = []
        self.per_epoch_learning_rates = []

    def on_train_batch_end(self, context: PhaseContext) -> None:
        self.per_step_learning_rates.append(context.optimizer.param_groups[0]["lr"])

    def on_train_loader_end(self, context: PhaseContext) -> None:
        self.per_epoch_learning_rates.append(context.optimizer.param_groups[0]["lr"])


class ExponentialWarmupLRCallback(LRCallbackBase):
    """
    LR scheduling callback for exponential warmup.
    LR grows exponentially from warmup_initial_lr to initial lr.
    When warmup_initial_lr is None- LR climb starts from 0.001
    """

    def __init__(self, **kwargs):
        super().__init__(Phase.TRAIN_EPOCH_START, **kwargs)
        self.warmup_initial_lr = self.training_params.warmup_initial_lr or 0.001
        warmup_epochs = self.training_params.lr_warmup_epochs
        lr_start = self.warmup_initial_lr
        lr_end = self.initial_lr
        self.c1 = (lr_end - lr_start) / (np.exp(warmup_epochs) - 1.0)
        self.c2 = (lr_start * np.exp(warmup_epochs) - lr_end) / (np.exp(warmup_epochs) - 1.0)

    def perform_scheduling(self, context):
        self.lr = self.c1 * np.exp(context.epoch) + self.c2
        self.update_lr(context.optimizer, context.epoch, None)

    def is_lr_scheduling_enabled(self, context):
        return self.training_params.lr_warmup_epochs >= context.epoch


class LRWarmupTest(unittest.TestCase):
    def test_lr_warmup(self):
        # Define Model
        net = LeNet()
        trainer = Trainer("lr_warmup_test")

        lrs = []
        phase_callbacks = [TestLRCallback(lr_placeholder=lrs)]

        train_params = {
            "max_epochs": 5,
            "lr_updates": [],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 3,
            "initial_lr": 1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": phase_callbacks,
            "warmup_mode": "linear_epoch_step",
        }

        expected_lrs = [0.25, 0.5, 0.75, 1.0, 1.0]
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4),
            valid_loader=classification_test_dataloader(batch_size=4),
        )
        self.assertListEqual(lrs, expected_lrs)

    def test_lr_warmup_with_lr_scheduling(self):
        # Define model
        net = LeNet()
        trainer = Trainer("lr_warmup_test")

        lrs = []
        phase_callbacks = [TestLRCallback(lr_placeholder=lrs)]

        train_params = {
            "max_epochs": 5,
            "cosine_final_lr_ratio": 0.2,
            "lr_mode": "cosine",
            "lr_warmup_epochs": 3,
            "initial_lr": 1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": phase_callbacks,
            "warmup_mode": "linear_epoch_step",
        }

        expected_lrs = [0.25, 0.5, 0.75, 0.9236067977499791, 0.4763932022500211]
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4, dataset_size=5),
            valid_loader=classification_test_dataloader(batch_size=4, dataset_size=5),
        )

        # ALTHOUGH NOT SEEN IN HERE, THE 4TH EPOCH USES LR=1, SO THIS IS THE EXPECTED LIST AS WE COLLECT
        # THE LRS AFTER THE UPDATE
        np.testing.assert_allclose(np.array(lrs), np.array(expected_lrs), rtol=1e-6)

    def test_warmup_linear_batch_step(self):
        # Define model
        net = LeNet()
        trainer = Trainer("lr_warmup_test_per_step")

        collect_lr_callback = CollectLRCallback()

        warmup_initial_lr = 0.05
        lr_warmup_steps = 100
        initial_lr = 1
        cosine_final_lr_ratio = 0.2
        max_epochs = 5

        train_params = {
            "max_epochs": max_epochs,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": cosine_final_lr_ratio,
            "warmup_initial_lr": warmup_initial_lr,
            "warmup_mode": "linear_batch_step",
            "lr_warmup_steps": lr_warmup_steps,
            "initial_lr": 1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": [collect_lr_callback],
        }

        train_loader = classification_test_dataloader(batch_size=4, dataset_size=1024)
        valid_loader = classification_test_dataloader(batch_size=4, dataset_size=5)

        expected_warmup_lrs = np.linspace(warmup_initial_lr, initial_lr, lr_warmup_steps).tolist()
        total_steps = max_epochs * len(train_loader) - lr_warmup_steps

        expected_cosine_lrs = CosineLRCallback.compute_learning_rate(
            step=np.arange(0, total_steps), total_steps=total_steps, initial_lr=initial_lr, final_lr_ratio=cosine_final_lr_ratio
        )

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=valid_loader,
        )

        np.testing.assert_allclose(collect_lr_callback.per_step_learning_rates[:100], expected_warmup_lrs, rtol=1e-4)
        np.testing.assert_allclose(collect_lr_callback.per_step_learning_rates[100:], expected_cosine_lrs, rtol=1e-4)

    def test_warmup_linear_epoch_step(self):
        # Define model
        net = LeNet()
        trainer = Trainer("test_warmup_initial_lr")

        collect_lr_callback = CollectLRCallback()

        train_params = {
            "max_epochs": 5,
            "lr_updates": [],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 3,
            "initial_lr": 1,
            "warmup_initial_lr": 4.0,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": [collect_lr_callback],
            "warmup_mode": "linear_epoch_step",
        }

        expected_lrs = [4.0, 3.0, 2.0, 1.0, 1.0]
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4, dataset_size=5),
            valid_loader=classification_test_dataloader(batch_size=4, dataset_size=5),
        )
        self.assertListEqual(collect_lr_callback.per_epoch_learning_rates, expected_lrs)

    def test_custom_lr_warmup(self):
        # Define model
        net = LeNet()
        trainer = Trainer("custom_lr_warmup_test")

        lrs = []
        phase_callbacks = [TestLRCallback(lr_placeholder=lrs)]

        train_params = {
            "max_epochs": 5,
            "lr_updates": [],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 3,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": phase_callbacks,
            "warmup_mode": ExponentialWarmupLRCallback,
            "initial_lr": 1.0,
            "warmup_initial_lr": 0.1,
        }

        expected_lrs = [0.1, 0.18102751585334242, 0.40128313980266034, 1.0, 1.0]
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4),
            valid_loader=classification_test_dataloader(batch_size=4),
        )
        np.testing.assert_allclose(np.array(lrs), np.array(expected_lrs), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
