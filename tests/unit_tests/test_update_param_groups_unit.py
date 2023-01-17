import unittest
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models import LeNet
from super_gradients.training.utils import HpmStruct, get_param
from super_gradients.training.utils.callbacks import TestLRCallback
import numpy as np


class TestNet(LeNet):
    """
    Toy test net with update_param_groups method that hard codes some lr.
    """

    def __init__(self):
        super(TestNet, self).__init__()

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        initial_lr = get_param(training_params, "initial_lr")
        for param_group in param_groups:
            param_group["lr"] = initial_lr * (epoch + 1)
        return param_groups


class UpdateParamGroupsTest(unittest.TestCase):
    def test_lr_scheduling_with_update_param_groups(self):
        # Define Model
        net = TestNet()
        trainer = Trainer("lr_warmup_test")

        lrs = []
        phase_callbacks = [TestLRCallback(lr_placeholder=lrs)]

        train_params = {
            "max_epochs": 3,
            "lr_mode": "step",
            "lr_updates": [0, 1, 2],
            "initial_lr": 0.1,
            "lr_decay_factor": 1,
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
        }

        expected_lrs = np.array([0.1, 0.2, 0.3])
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        self.assertTrue(np.allclose(np.array(lrs), expected_lrs, rtol=0.0000001))
