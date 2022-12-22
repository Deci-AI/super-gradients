import unittest
import os
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.common.object_names import Models


class SaveCkptListUnitTest(unittest.TestCase):
    def setUp(self):
        # Define Parameters
        train_params = {
            "max_epochs": 4,
            "lr_decay_factor": 0.1,
            "lr_updates": [4],
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "save_ckpt_epoch_list": [1, 3],
            "loss": "cross_entropy",
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }

        # Define Model
        trainer = Trainer("save_ckpt_test")

        # Build Model
        model = models.get(Models.RESNET18_CIFAR, arch_params={"num_classes": 10})

        # Train Model (and save ckpt_epoch_list)
        trainer.train(model=model, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        dir_path = trainer.checkpoints_dir_path
        self.file_names_list = [dir_path + f"/ckpt_epoch_{epoch}.pth" for epoch in train_params["save_ckpt_epoch_list"]]

    def test_save_ckpt_epoch_list(self):
        self.assertTrue(os.path.exists(self.file_names_list[0]))
        self.assertTrue(os.path.exists(self.file_names_list[1]))


if __name__ == "__main__":
    unittest.main()
