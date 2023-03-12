import os
import unittest
from copy import deepcopy

from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.kd_trainer.kd_trainer import KDTrainer
import torch

from super_gradients.training import models
from super_gradients.training.losses.kd_losses import KDLogitsLoss
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models.classification_models.resnet import ResNet50, ResNet18
from super_gradients.training.models.kd_modules.kd_module import KDModule
from super_gradients.training.utils.callbacks import PhaseCallback, PhaseContext, Phase
from super_gradients.modules.utils import NormalizationAdapter
from super_gradients.training.utils.utils import check_models_have_same_weights


class PreTrainingNetCollector(PhaseCallback):
    def __init__(self):
        super(PreTrainingNetCollector, self).__init__(phase=Phase.PRE_TRAINING)
        self.net = None

    def __call__(self, context: PhaseContext):
        self.net = deepcopy(context.net)


class PreTrainingEMANetCollector(PhaseCallback):
    def __init__(self):
        super(PreTrainingEMANetCollector, self).__init__(phase=Phase.PRE_TRAINING)
        self.net = None

    def __call__(self, context: PhaseContext):
        self.net = deepcopy(context.ema_model)


class KDTrainerTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
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
        }

    def test_teacher_sg_module_methods(self):
        student = models.get(Models.RESNET18, arch_params={"num_classes": 1000})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 1000}, pretrained_weights="imagenet")
        kd_module = KDModule(arch_params={}, student=student, teacher=teacher)

        initial_param_groups = kd_module.initialize_param_groups(lr=0.1, training_params={})
        updated_param_groups = kd_module.update_param_groups(param_groups=initial_param_groups, lr=0.2, epoch=0, iter=0, training_params={}, total_batch=None)

        self.assertTrue(initial_param_groups[0]["lr"] == 0.2 == updated_param_groups[0]["lr"])

    def test_train_kd_module_external_models(self):
        sg_model = KDTrainer("test_train_kd_module_external_models")
        teacher_model = ResNet50(arch_params={}, num_classes=5)
        student_model = ResNet18(arch_params={}, num_classes=5)

        sg_model.train(
            training_params=self.kd_train_params,
            student=deepcopy(student_model),
            teacher=teacher_model,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )

        # TEACHER WEIGHT'S SHOULD REMAIN THE SAME
        self.assertTrue(check_models_have_same_weights(teacher_model, sg_model.net.module.teacher))

        # STUDENT WEIGHT'S SHOULD NOT REMAIN THE SAME
        self.assertFalse(check_models_have_same_weights(student_model, sg_model.net.module.student))

    def test_train_model_with_input_adapter(self):
        kd_trainer = KDTrainer("train_kd_module_with_with_input_adapter")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 5}, pretrained_weights="imagenet")

        adapter = NormalizationAdapter(
            mean_original=[0.485, 0.456, 0.406], std_original=[0.229, 0.224, 0.225], mean_required=[0.5, 0.5, 0.5], std_required=[0.5, 0.5, 0.5]
        )

        kd_arch_params = {"teacher_input_adapter": adapter}
        kd_trainer.train(
            training_params=self.kd_train_params,
            student=student,
            teacher=teacher,
            kd_arch_params=kd_arch_params,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )

        self.assertEqual(kd_trainer.net.module.teacher_input_adapter, adapter)

    def test_load_ckpt_best_for_student(self):
        kd_trainer = KDTrainer("test_load_ckpt_best")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 5}, pretrained_weights="imagenet")
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        kd_trainer.train(
            training_params=train_params,
            student=student,
            teacher=teacher,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )
        best_student_ckpt = os.path.join(kd_trainer.checkpoints_dir_path, "ckpt_best.pth")

        student_reloaded = models.get(Models.RESNET18, arch_params={"num_classes": 5}, checkpoint_path=best_student_ckpt)

        self.assertTrue(check_models_have_same_weights(student_reloaded, kd_trainer.net.module.student))

    def test_load_ckpt_best_for_student_with_ema(self):
        kd_trainer = KDTrainer("test_load_ckpt_best")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 5}, pretrained_weights="imagenet")
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        train_params["ema"] = True
        train_params["ema_params"] = {"decay_type": "constant", "decay": 0.999}

        kd_trainer.train(
            training_params=train_params,
            student=student,
            teacher=teacher,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )
        best_student_ckpt = os.path.join(kd_trainer.checkpoints_dir_path, "ckpt_best.pth")

        student_reloaded = models.get(Models.RESNET18, arch_params={"num_classes": 5}, checkpoint_path=best_student_ckpt)

        self.assertTrue(check_models_have_same_weights(student_reloaded, kd_trainer.ema_model.ema.module.student))

    def test_resume_kd_training(self):
        kd_trainer = KDTrainer("test_resume_training_start")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 5}, pretrained_weights="imagenet")
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        kd_trainer.train(
            training_params=train_params,
            student=student,
            teacher=teacher,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )
        latest_net = deepcopy(kd_trainer.net)

        kd_trainer = KDTrainer("test_resume_training_start")
        student = models.get(Models.RESNET18, arch_params={"num_classes": 5})
        teacher = models.get(Models.RESNET50, arch_params={"num_classes": 5}, pretrained_weights="imagenet")

        train_params["max_epochs"] = 2
        train_params["resume"] = True
        collector = PreTrainingNetCollector()
        train_params["phase_callbacks"] = [collector]
        kd_trainer.train(
            training_params=train_params,
            student=student,
            teacher=teacher,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(collector.net, latest_net))


if __name__ == "__main__":
    unittest.main()
