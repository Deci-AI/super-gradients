import os
import unittest
from copy import deepcopy

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.models import LeNet


class FirstEpochInfoCollector(PhaseCallback):
    def __init__(self):
        super().__init__(phase=Phase.TRAIN_EPOCH_START)
        self.called = False
        self.first_epoch = None
        self.first_epoch_net = None

    def __call__(self, context: PhaseContext):
        if not self.called:
            self.first_epoch = context.epoch
            self.first_epoch_net = deepcopy(context.net)
            self.called = True


class ResumeTrainingTest(unittest.TestCase):
    def test_resume_training(self):
        train_params = {
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
            "greater_metric_to_watch_is_better": True,
        }

        # Define Model
        net = LeNet()
        trainer = Trainer("test_resume_training")
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 3 AND THE END OF EPOCH NUMBER 2
        resume_net = LeNet()
        trainer = Trainer("test_resume_training")
        first_epoch_cb = FirstEpochInfoCollector()
        train_params["resume"] = True
        train_params["max_epochs"] = 3
        train_params["phase_callbacks"] = [first_epoch_cb]
        trainer.train(
            model=resume_net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        # ASSERT RELOADED MODEL HAS THE SAME WEIGHTS AS THE MODEL SAVED IN FIRST PART OF TRAINING
        self.assertTrue(check_models_have_same_weights(net, first_epoch_cb.first_epoch_net))

        # ASSERT WE START FROM THE RIGHT EPOCH NUMBER
        self.assertTrue(first_epoch_cb.first_epoch == 2)

    def test_resume_external_training(self):
        train_params = {
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
            "greater_metric_to_watch_is_better": True,
        }
        # Define Model
        net = LeNet()
        trainer = Trainer("test_resume_training")
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 3 AND THE END OF EPOCH NUMBER 2
        resume_net = LeNet()
        resume_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_latest.pth")

        # SET DIFFERENT EXPERIMENT NAME SO WE LOAD A CHECKPOINT THAT HAS A DIFFERENT PATH FROM THE DEFAULT ONE
        trainer = Trainer("test_resume_external_training")
        first_epoch_cb = FirstEpochInfoCollector()
        train_params["resume_path"] = resume_path
        train_params["max_epochs"] = 3
        train_params["phase_callbacks"] = [first_epoch_cb]
        trainer.train(
            model=resume_net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        # ASSERT RELOADED MODEL HAS THE SAME WEIGHTS AS THE MODEL SAVED IN FIRST PART OF TRAINING
        self.assertTrue(check_models_have_same_weights(net, first_epoch_cb.first_epoch_net))

        # ASSERT WE START FROM THE RIGHT EPOCH NUMBER
        self.assertTrue(first_epoch_cb.first_epoch == 2)


if __name__ == "__main__":
    unittest.main()
