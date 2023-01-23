import unittest

from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils.utils import check_models_have_same_weights
import os


class LocalCkptHeadReplacementTest(unittest.TestCase):
    def test_local_ckpt_head_replacement(self):
        train_params = {
            "max_epochs": 1,
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
        net = models.get(Models.RESNET18, num_classes=5)
        trainer = Trainer("test_resume_training")
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())
        ckpt_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_latest.pth")

        net2 = models.get(Models.RESNET18, num_classes=10, checkpoint_num_classes=5, checkpoint_path=ckpt_path)
        self.assertFalse(check_models_have_same_weights(net, net2))

        net.linear = None
        net2.linear = None
        self.assertTrue(check_models_have_same_weights(net, net2))
