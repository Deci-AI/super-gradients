import shutil
import unittest

from super_gradients.common.object_names import Models
from super_gradients.training import models

import super_gradients
import torch
import os
from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super_gradients.init_trainer()
        # NAMES FOR THE EXPERIMENTS TO LATER DELETE
        cls.experiment_names = ["test_train", "test_save_load", "test_load_w", "test_load_w2", "test_load_w3", "test_checkpoint_content", "analyze"]
        cls.training_params = {
            "max_epochs": 1,
            "silent_mode": True,
            "lr_decay_factor": 0.1,
            "initial_lr": 0.1,
            "lr_updates": [4],
            "lr_mode": "step",
            "loss": "cross_entropy",
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }

    @classmethod
    def tearDownClass(cls) -> None:
        # ERASE ALL THE EXPERIMENT FOLDERS THAT WERE CREATED DURING THIS TEST
        for experiment_name in cls.experiment_names:
            experiment_dir = get_checkpoints_dir_path(experiment_name=experiment_name)
            if os.path.isdir(experiment_dir):
                shutil.rmtree(experiment_dir)

    @staticmethod
    def get_classification_trainer(name=""):
        trainer = Trainer(name)
        model = models.get(Models.RESNET18, num_classes=5)
        return trainer, model

    def test_train(self):
        trainer, model = self.get_classification_trainer(self.experiment_names[0])
        trainer.train(
            model=model, training_params=self.training_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

    def test_save_load(self):
        trainer, model = self.get_classification_trainer(self.experiment_names[1])
        trainer.train(
            model=model, training_params=self.training_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )
        resume_training_params = self.training_params.copy()
        resume_training_params["resume"] = True
        resume_training_params["max_epochs"] = 2
        trainer, model = self.get_classification_trainer(self.experiment_names[1])
        trainer.train(
            model=model, training_params=resume_training_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

    def test_checkpoint_content(self):
        """VERIFY THAT ALL CHECKPOINTS ARE SAVED AND CONTAIN ALL THE EXPECTED KEYS"""
        trainer, model = self.get_classification_trainer(self.experiment_names[5])
        params = self.training_params.copy()
        params["save_ckpt_epoch_list"] = [1]
        trainer.train(model=model, training_params=params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())
        ckpt_filename = ["ckpt_best.pth", "ckpt_latest.pth", "ckpt_epoch_1.pth"]
        ckpt_paths = [os.path.join(trainer.checkpoints_dir_path, suf) for suf in ckpt_filename]
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path)
            self.assertListEqual(["net", "acc", "epoch", "optimizer_state_dict", "scaler_state_dict"], list(ckpt.keys()))
        trainer._save_checkpoint()
        weights_only = torch.load(os.path.join(trainer.checkpoints_dir_path, "ckpt_latest_weights_only.pth"))
        self.assertListEqual(["net"], list(weights_only.keys()))


if __name__ == "__main__":
    unittest.main()
