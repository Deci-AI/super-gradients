import unittest

import torch

from super_gradients import ClassificationTestDatasetInterface, SgModel
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18


class FactoriesTest(unittest.TestCase):

    def test_training_with_factories(self):
        model = SgModel("test_train_with_factories", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = {"classification_test_dataset": {"dataset_params": dataset_params}}
        model.connect_dataset_interface(dataset)

        net = ResNet18(num_classes=5, arch_params={})
        model.build_model(net)
        train_params = {"max_epochs": 2,
                        "lr_updates": [1],
                        "lr_decay_factor": 0.1,
                        "lr_mode": "step",
                        "lr_warmup_epochs": 0,
                        "initial_lr": 0.1,
                        "loss": "cross_entropy",
                        "optimizer": "torch.optim.ASGD",    # use an optimizer by factory
                        "criterion_params": {},
                        "optimizer_params": {"lambd": 0.0001, "alpha": 0.75},
                        "train_metrics_list": ["Accuracy", "Top5"],  # use a metric by factory
                        "valid_metrics_list": ["Accuracy", "Top5"],  # use a metric by factory
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}

        model.train(train_params)

        self.assertIsInstance(model.train_metrics.Accuracy, Accuracy)
        self.assertIsInstance(model.valid_metrics.Top5, Top5)
        self.assertIsInstance(model.dataset_interface, ClassificationTestDatasetInterface)
        self.assertIsInstance(model.optimizer, torch.optim.ASGD)

if __name__ == '__main__':
    unittest.main()
