import torch
import torch.nn as nn
import unittest

from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.models.classification_models.resnet import ResNet18
from super_gradients.training.metrics import Accuracy, Top5
from torchmetrics.metric import Metric


class MetricTest(Metric):
    def __init__(self, metric_values):
        super().__init__()
        self.metrics_values = metric_values
        self.count = 0

    def update(self, *args, **kwargs) -> None:
        pass

    def compute(self):
        value = self.metrics_values[self.count]
        self.count += 1
        return value


class LossTest(nn.Module):
    def __init__(self, loss_values):
        super(LossTest, self).__init__()
        self.loss_values = loss_values
        self.count = 0

    def forward(self, pred, label):
        # double the loss values, one step for training and one for validation
        # make returned loss differentiable
        loss = (pred * 0).sum() + self.loss_values[self.count // 2]
        self.count += 1
        return loss, torch.stack([loss]).detach()


class EarlyStopTest(unittest.TestCase):
    def setUp(self) -> None:
        # batch_size is equal to length of dataset, to have only one step per epoch, to ease the test.
        self.net = ResNet18(num_classes=5, arch_params={})
        self.max_epochs = 10
        self.train_params = {
            "max_epochs": self.max_epochs,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Top5()],
            "metric_to_watch": "Top5",
            "greater_metric_to_watch_is_better": True,
            "average_best_models": False,
        }

    def test_min_mode_patience_metric(self):
        """
        Test for mode=min metric, test that training stops after no improvement in metric value for amount of `patience`
        epochs.
        """
        trainer = Trainer("early_stop_test")

        early_stop_loss = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="LossTest", mode="min", patience=3, verbose=True)
        phase_callbacks = [early_stop_loss]

        loss_values = torch.tensor([1.0, 0.8, 0.81, 0.8, 0.9, 0.2, 0.1, 0.3, 0.05, 0.9])
        fake_loss = LossTest(loss_values)
        train_params = self.train_params.copy()
        train_params.update({"loss": fake_loss, "phase_callbacks": phase_callbacks})

        trainer.train(
            model=self.net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )
        excepted_end_epoch = 5

        # count divided by 2, because loss counter used for both train and eval.
        self.assertEqual(excepted_end_epoch, fake_loss.count // 2)

    def test_max_mode_patience_metric(self):
        """
        Test for mode=max metric, test that training stops after no improvement in metric value for amount of `patience`
        epochs.
        """
        trainer = Trainer("early_stop_test")
        early_stop_acc = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="MetricTest", mode="max", patience=3, verbose=True)
        phase_callbacks = [early_stop_acc]

        metric_values = torch.tensor([0.2, 0.1, 0.3, 0.28, 0.2, 0.1, 0.33, 0.05, 0.9, 0.99])
        fake_metric = MetricTest(metric_values)
        train_params = self.train_params.copy()
        train_params.update({"valid_metrics_list": [fake_metric], "metric_to_watch": "MetricTest", "phase_callbacks": phase_callbacks})

        trainer.train(
            model=self.net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )
        excepted_end_epoch = 6

        self.assertEqual(excepted_end_epoch, fake_metric.count)

    def test_min_mode_threshold_metric(self):
        """
        Test for mode=min metric, test that training stops after metric value reaches the `threshold` value.
        """
        trainer = Trainer("early_stop_test")

        early_stop_loss = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="LossTest", mode="min", threshold=0.1, verbose=True)
        phase_callbacks = [early_stop_loss]

        loss_values = torch.tensor([1.0, 0.8, 0.4, 0.2, 0.09, 0.11, 0.105, 0.3, 0.05, 0.02])
        fake_loss = LossTest(loss_values)
        train_params = self.train_params.copy()
        train_params.update({"loss": fake_loss, "phase_callbacks": phase_callbacks})

        trainer.train(
            model=self.net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )
        excepted_end_epoch = 5
        # count divided by 2, because loss counter used for both train and eval.
        self.assertEqual(excepted_end_epoch, fake_loss.count // 2)

    def test_max_mode_threshold_metric(self):
        """
        Test for mode=max metric, test that training stops after metric value reaches the `threshold` value.
        """
        trainer = Trainer("early_stop_test")

        early_stop_acc = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="MetricTest", mode="max", threshold=0.94, verbose=True)
        phase_callbacks = [early_stop_acc]

        metric_values = torch.tensor([0.2, 0.1, 0.6, 0.8, 0.9, 0.92, 0.95, 0.94, 0.948, 0.99])
        fake_metric = MetricTest(metric_values)
        train_params = self.train_params.copy()
        train_params.update({"valid_metrics_list": [fake_metric], "metric_to_watch": "MetricTest", "phase_callbacks": phase_callbacks})

        trainer.train(
            model=self.net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )
        excepted_end_epoch = 7

        self.assertEqual(excepted_end_epoch, fake_metric.count)

    def test_no_finite_stoppage(self):
        """
        Test that training stops when monitor value is not a finite number. Test case of NaN and Inf values.
        """
        # test Nan value
        trainer = Trainer("early_stop_test")

        early_stop_loss = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="LossTest", mode="min", check_finite=True, verbose=True)
        phase_callbacks = [early_stop_loss]

        loss_values = torch.tensor([1.0, float("nan"), 0.81, 0.8, 0.9, 0.2, 0.1, 0.3, 0.05, 0.9])
        fake_loss = LossTest(loss_values)
        train_params = self.train_params.copy()
        train_params.update({"loss": fake_loss, "phase_callbacks": phase_callbacks})

        trainer.train(
            model=self.net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )
        excepted_end_epoch = 2

        self.assertEqual(excepted_end_epoch, fake_loss.count // 2)

        # test Inf value
        trainer = Trainer("early_stop_test")

        early_stop_loss = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="LossTest", mode="min", patience=3, verbose=True)
        phase_callbacks = [early_stop_loss]

        loss_values = torch.tensor([1.0, 0.8, float("inf"), 0.8, 0.9, 0.2, 0.1, 0.3, 0.05, 0.9])
        fake_loss = LossTest(loss_values)
        train_params = self.train_params.copy()
        train_params.update({"loss": fake_loss, "phase_callbacks": phase_callbacks})

        trainer.train(
            model=self.net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )
        excepted_end_epoch = 3
        # count divided by 2, because loss counter used for both train and eval.
        self.assertEqual(excepted_end_epoch, fake_loss.count // 2)

    def test_min_delta(self):
        """
        Test for `min_delta` argument, metric value is considered an improvement only if
        current_value - min_delta > best_value
        """
        trainer = Trainer("early_stop_test")

        early_stop_acc = EarlyStop(Phase.VALIDATION_EPOCH_END, monitor="MetricTest", mode="max", patience=2, min_delta=0.1, verbose=True)
        phase_callbacks = [early_stop_acc]

        metric_values = torch.tensor([0.1, 0.2, 0.305, 0.31, 0.34, 0.42, 0.6, 0.8, 0.9, 0.99])
        fake_metric = MetricTest(metric_values)
        train_params = self.train_params.copy()
        train_params.update({"valid_metrics_list": [fake_metric], "metric_to_watch": "MetricTest", "phase_callbacks": phase_callbacks})

        trainer.train(
            model=self.net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        excepted_end_epoch = 5

        self.assertEqual(excepted_end_epoch, fake_metric.count)


if __name__ == "__main__":
    unittest.main()
