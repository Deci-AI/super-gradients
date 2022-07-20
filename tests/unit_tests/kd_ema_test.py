import unittest
from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.kd_trainer.kd_trainer import KDTrainer
import torch
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy
from super_gradients.training.losses.kd_losses import KDLogitsLoss


class KDEMATest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sg_trained_teacher = Trainer("sg_trained_teacher", device='cpu')
        cls.dataset_params = {"batch_size": 5}
        cls.dataset = ClassificationTestDatasetInterface(dataset_params=cls.dataset_params)

        cls.kd_train_params = {"max_epochs": 3, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                               "lr_warmup_epochs": 0, "initial_lr": 0.1,
                               "loss": KDLogitsLoss(torch.nn.CrossEntropyLoss()),
                               "optimizer": "SGD",
                               "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                               "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                               "metric_to_watch": "Accuracy",
                               'loss_logging_items_names': ["Loss", "Task Loss", "Distillation Loss"],
                               "greater_metric_to_watch_is_better": True, "average_best_models": False,
                               "ema": True}

    def test_teacher_ema_not_duplicated(self):
        """Check that the teacher EMA is a reference to the teacher net (not a copy)."""

        kd_trainer = KDTrainer("test_teacher_ema_not_duplicated", device='cpu')
        kd_trainer.connect_dataset_interface(self.dataset)
        kd_trainer.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                             run_teacher_on_eval=True, )

        kd_trainer.train(self.kd_train_params)

        self.assertTrue(kd_trainer.ema_model.ema.module.teacher is kd_trainer.net.module.teacher)
        self.assertTrue(kd_trainer.ema_model.ema.module.student is not kd_trainer.net.module.student)

    def test_kd_ckpt_reload_ema(self):
        """Check that the KD model load correctly from checkpoint when "load_ema_as_net=True"."""

        # Create a KD model and train it
        kd_trainer = KDTrainer("test_kd_ema_ckpt_reload", device='cpu')
        kd_trainer.connect_dataset_interface(self.dataset)
        kd_trainer.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                             run_teacher_on_eval=True, )
        kd_trainer.train(self.kd_train_params)
        ema_model = kd_trainer.ema_model.ema
        net = kd_trainer.net

        # Load the trained KD model
        kd_trainer = KDTrainer("test_kd_ema_ckpt_reload", device='cpu')
        kd_trainer.connect_dataset_interface(self.dataset)
        kd_trainer.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={"load_checkpoint": True, "load_ema_as_net": True},
                             run_teacher_on_eval=True, )

        # TRAIN FOR 0 EPOCHS JUST TO SEE THAT WHEN CONTINUING TRAINING EMA MODEL HAS BEEN SAVED CORRECTLY
        kd_trainer.train(self.kd_train_params)
        reloaded_ema_model = kd_trainer.ema_model.ema
        reloaded_net = kd_trainer.net

        # trained ema == loaded ema (Should always be true as long as "ema=True" in train_params)
        self.assertTrue(check_models_have_same_weights(ema_model, reloaded_ema_model))

        # loaded net != trained net (since load_ema_as_net = True)
        self.assertTrue(not check_models_have_same_weights(reloaded_net, net))

        # loaded net == trained ema (since load_ema_as_net = True)
        self.assertTrue(check_models_have_same_weights(reloaded_net, ema_model))

        # loaded student ema == loaded student net (since load_ema_as_net = True)
        self.assertTrue(check_models_have_same_weights(reloaded_ema_model.module.student, reloaded_net.module.student))

        # loaded teacher ema == loaded teacher net (teacher always loads ema)
        self.assertTrue(check_models_have_same_weights(reloaded_ema_model.module.teacher, reloaded_net.module.teacher))

    def test_kd_ckpt_reload_net(self):
        """Check that the KD model load correctly from checkpoint when "load_ema_as_net=False"."""

        # Create a KD model and train it
        kd_trainer = KDTrainer("test_kd_ema_ckpt_reload", device='cpu')
        kd_trainer.connect_dataset_interface(self.dataset)
        kd_trainer.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                             run_teacher_on_eval=True, )
        kd_trainer.train(self.kd_train_params)
        ema_model = kd_trainer.ema_model.ema
        net = kd_trainer.net

        # Load the trained KD model
        kd_trainer = KDTrainer("test_kd_ema_ckpt_reload", device='cpu')
        kd_trainer.connect_dataset_interface(self.dataset)
        kd_trainer.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={"load_checkpoint": True, "load_ema_as_net": False},
                             run_teacher_on_eval=True, )

        # TRAIN FOR 0 EPOCHS JUST TO SEE THAT WHEN CONTINUING TRAINING EMA MODEL HAS BEEN SAVED CORRECTLY
        kd_trainer.train(self.kd_train_params)
        reloaded_ema_model = kd_trainer.ema_model.ema
        reloaded_net = kd_trainer.net

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


if __name__ == '__main__':
    unittest.main()
