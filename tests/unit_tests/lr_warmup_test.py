import unittest

import numpy as np

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models import LeNet
from super_gradients.training.utils.callbacks import TestLRCallback, LRCallbackBase, Phase


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
            "warmup_mode": "linear_step",
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
            "warmup_mode": "linear_step",
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
        self.assertListEqual(lrs, expected_lrs)

    def test_warmup_initial_lr(self):
        # Define model
        net = LeNet()
        trainer = Trainer("test_warmup_initial_lr")

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
            "warmup_mode": "linear_step",
            "initial_lr": 1,
            "warmup_initial_lr": 4.0,
        }

        expected_lrs = [4.0, 3.0, 2.0, 1.0, 1.0]
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4, dataset_size=5),
            valid_loader=classification_test_dataloader(batch_size=4, dataset_size=5),
        )
        self.assertListEqual(lrs, expected_lrs)

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
        self.assertListEqual(lrs, expected_lrs)


if __name__ == "__main__":
    unittest.main()
