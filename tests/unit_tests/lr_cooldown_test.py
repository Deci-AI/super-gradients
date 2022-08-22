import unittest
from super_gradients.training import Trainer
from super_gradients.training.metrics import Accuracy
from super_gradients.training.datasets import ClassificationTestDatasetInterface
from super_gradients.training.models import LeNet
from super_gradients.training.utils.callbacks import TestLRCallback


class LRCooldownTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_params = {"batch_size": 4}
        self.dataset = ClassificationTestDatasetInterface(dataset_params=self.dataset_params)
        self.arch_params = {'num_classes': 10}

    def test_lr_cooldown_with_lr_scheduling(self):
        # Define Model
        net = LeNet()
        trainer = Trainer("lr_warmup_test", model_checkpoints_location='local')
        trainer.connect_dataset_interface(self.dataset)

        lrs = []
        phase_callbacks = [TestLRCallback(lr_placeholder=lrs)]

        train_params = {"max_epochs": 7, "cosine_final_lr_ratio": 0.2, "lr_mode": "cosine",
                        "lr_cooldown_epochs": 2,
                        "lr_warmup_epochs": 3, "initial_lr": 1, "loss": "cross_entropy", "optimizer": 'SGD',
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True, "ema": False, "phase_callbacks": phase_callbacks}

        expected_lrs = [0.25, 0.5, 0.75, 0.9236067977499791, 0.4763932022500211, 0.4763932022500211, 0.4763932022500211]
        trainer.train(net=net, training_params=train_params)

        # ALTHOUGH NOT SEEN IN HERE, THE 4TH EPOCH USES LR=1, SO THIS IS THE EXPECTED LIST AS WE COLLECT
        # THE LRS AFTER THE UPDATE
        self.assertListEqual(lrs, expected_lrs)
