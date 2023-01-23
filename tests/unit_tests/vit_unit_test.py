import unittest

from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients import Trainer
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training import models


class TestViT(unittest.TestCase):
    def setUp(self):
        self.arch_params = {"image_size": (224, 224), "patch_size": (16, 16), "num_classes": 10}

        self.train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
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
        }

    def test_train_vit(self):
        """
        Validate vit_base
        """
        trainer = Trainer("test_vit_base")
        model = models.get(Models.VIT_BASE, arch_params=self.arch_params, num_classes=5)
        trainer.train(
            model=model, training_params=self.train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )


if __name__ == "__main__":
    unittest.main()
