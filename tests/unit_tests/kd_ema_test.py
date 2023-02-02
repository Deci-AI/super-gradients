import unittest

from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.kd_trainer import KDTrainer
import torch
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.metrics import Accuracy
from super_gradients.training.losses.kd_losses import KDLogitsLoss
from super_gradients.common.object_names import Models


class KDEMATest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sg_trained_teacher = Trainer("sg_trained_teacher")

        cls.kd_train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": KDLogitsLoss(torch.nn.CrossEntropyLoss()),
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "loss_logging_items_names": ["Loss", "Task Loss", "Distillation Loss"],
            "greater_metric_to_watch_is_better": True,
            "average_best_models": False,
            "ema": True,
            "ema_params": {"decay_type": "constant", "decay": 0.999},
        }

    def test_teacher_ema_not_duplicated(self):
        """Check that the teacher EMA is a reference to the teacher net (not a copy)."""

        kd_model = KDTrainer("test_teacher_ema_not_duplicated")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 1000})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 1000}, pretrained_weights="imagenet")

        kd_model.train(
            training_params=self.kd_train_params,
            student=student,
            teacher=teacher,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )

        self.assertTrue(kd_model.ema_model.ema.module.teacher is kd_model.net.module.teacher)
        self.assertTrue(kd_model.ema_model.ema.module.student is not kd_model.net.module.student)

    def test_kd_ckpt_reload_net(self):
        """Check that the KD trainer load correctly from checkpoint when "load_ema_as_net=False"."""

        # Create a KD trainer and train it
        train_params = self.kd_train_params.copy()
        kd_model = KDTrainer("test_kd_ema_ckpt_reload")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 1000})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 1000}, pretrained_weights="imagenet")

        kd_model.train(
            training_params=self.kd_train_params,
            student=student,
            teacher=teacher,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )
        ema_model = kd_model.ema_model.ema
        net = kd_model.net

        # Load the trained KD trainer
        kd_model = KDTrainer("test_kd_ema_ckpt_reload")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 1000})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 1000}, pretrained_weights="imagenet")

        train_params["resume"] = True
        kd_model.train(
            training_params=train_params,
            student=student,
            teacher=teacher,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )
        reloaded_ema_model = kd_model.ema_model.ema
        reloaded_net = kd_model.net

        # trained ema == loaded ema (Should always be true as long as "ema=True" in train_params)
        self.assertTrue(check_models_have_same_weights(ema_model, reloaded_ema_model))

        # loaded net == trained net (since load_ema_as_net = False)
        self.assertTrue(check_models_have_same_weights(reloaded_net, net))

        # loaded net != trained ema (since load_ema_as_net = False)
        self.assertTrue(not check_models_have_same_weights(reloaded_net, ema_model))

        # loaded student ema == loaded  student net (since load_ema_as_net = False)
        self.assertTrue(not check_models_have_same_weights(reloaded_ema_model.module.student, reloaded_net.module.student))

        # loaded teacher ema == loaded teacher net (teacher always loads ema)
        self.assertTrue(check_models_have_same_weights(reloaded_ema_model.module.teacher, reloaded_net.module.teacher))


if __name__ == "__main__":
    unittest.main()
