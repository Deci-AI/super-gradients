import unittest
import os
from super_gradients.training.sg_model import SgModel
from super_gradients.training.kd_model.kd_model import KDModel
import torch
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy
from super_gradients.training.exceptions.kd_model_exceptions import ArchitectureKwargsException, \
    UnsupportedKDArchitectureException, InconsistentParamsException, UnsupportedKDModelArgException, \
    TeacherKnowledgeException
from super_gradients.training.models.classification_models.resnet import ResNet50, ResNet18
from super_gradients.training.losses.kd_losses import KDLogitsLoss
from copy import deepcopy
from super_gradients.training.utils.module_utils import NormalizationAdapter


class KDModelTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sg_trained_teacher = SgModel("sg_trained_teacher", device='cpu')
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

        kd_model = KDModel("test_teacher_ema_not_duplicated", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        kd_model.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                             run_teacher_on_eval=True, )

        kd_model.train(self.kd_train_params)

        assert kd_model.ema_model.ema.module.teacher is kd_model.net.module.teacher

    def test_kd_ckpt_reload_ema(self):
        """Check that the KD model load correctly from checkpoint when "load_ema_as_net=True"."""

        kd_model = KDModel("test_kd_ema_ckpt_reload", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        kd_model.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                             run_teacher_on_eval=True, )

        kd_model.train(self.kd_train_params)
        ema_model = kd_model.ema_model.ema
        net = kd_model.net

        kd_model = KDModel("test_kd_ema_ckpt_reload", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        kd_model.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={"load_checkpoint": True, "load_ema_as_net": True},
                             run_teacher_on_eval=True, )

        # TRAIN FOR 0 EPOCHS JUST TO SEE THAT WHEN CONTINUING TRAINING EMA MODEL HAS BEEN SAVED CORRECTLY
        kd_model.train(self.kd_train_params)
        reloaded_ema_model = kd_model.ema_model.ema
        reloaded_net = kd_model.net

        # trained ema == loaded ema (Should always be true as long as "ema=True" in train_params)
        assert check_models_have_same_weights(ema_model, reloaded_ema_model)

        # trained net != loaded net (since load_ema_as_net = True)
        assert not check_models_have_same_weights(net, reloaded_net)

        # trained ema != loaded net (since load_ema_as_net = True)
        assert check_models_have_same_weights(ema_model, reloaded_net)

        # loaded student ema == loaded  student net (since load_ema_as_net = True)
        assert check_models_have_same_weights(reloaded_ema_model.module.student, reloaded_net.module.student)

        # loaded teacher ema == loaded teacher net (teacher always loads ema)
        assert check_models_have_same_weights(reloaded_ema_model.module.teacher, reloaded_net.module.teacher)



    def test_kd_ckpt_reload_net(self):
        """Check that the KD model load correctly from checkpoint when "load_ema_as_net=False"."""

        kd_model = KDModel("test_kd_ema_ckpt_reload", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        kd_model.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                             run_teacher_on_eval=True, )


        kd_model.train(self.kd_train_params)
        ema_model = kd_model.ema_model.ema
        net = kd_model.net

        kd_model = KDModel("test_kd_ema_ckpt_reload", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        kd_model.build_model(student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000},
                             teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={"load_checkpoint": True, "load_ema_as_net": False},
                             run_teacher_on_eval=True, )

        # TRAIN FOR 0 EPOCHS JUST TO SEE THAT WHEN CONTINUING TRAINING EMA MODEL HAS BEEN SAVED CORRECTLY
        kd_model.train(self.kd_train_params)
        reloaded_ema_model = kd_model.ema_model.ema
        reloaded_net = kd_model.net

        # trained ema == loaded ema (Should always be true as long as "ema=True" in train_params)
        assert check_models_have_same_weights(ema_model, reloaded_ema_model)

        # trained net == loaded net (since load_ema_as_net = False)
        assert check_models_have_same_weights(net, reloaded_net)

        # trained ema != loaded net (since load_ema_as_net = False)
        assert not check_models_have_same_weights(ema_model, reloaded_net)

        # loaded student ema == loaded  student net (since load_ema_as_net = False)
        assert not check_models_have_same_weights(reloaded_ema_model.module.student, reloaded_net.module.student)

        # loaded teacher ema == loaded teacher net (teacher always loads ema)
        assert check_models_have_same_weights(reloaded_ema_model.module.teacher, reloaded_net.module.teacher)


if __name__ == '__main__':
    unittest.main()
