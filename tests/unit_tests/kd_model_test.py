import unittest
import os

from super_gradients.training import models
from super_gradients.training.models.kd_modules.kd_module import KDModule

from super_gradients.training.sg_model import SgModel
from super_gradients.training.kd_model.kd_model import KDModel
import torch

from super_gradients.training.utils.callbacks import PhaseCallback, PhaseContext, Phase
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models.classification_models.resnet import ResNet50, ResNet18
from super_gradients.training.losses.kd_losses import KDLogitsLoss
from copy import deepcopy
from super_gradients.training.utils.module_utils import NormalizationAdapter


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


class KDModelTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
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
                               "greater_metric_to_watch_is_better": True, "average_best_models": False}

    def test_teacher_sg_module_methods(self):
        student = models.get('resnet18', arch_params={'num_classes': 1000})
        teacher = models.get('resnet50', arch_params={'num_classes': 1000},
                             checkpoint_params={"pretrained_weights": "imagenet"})
        kd_module = KDModule(arch_params={},
                             student=student,
                             teacher=teacher
                             )

        initial_param_groups = kd_module.initialize_param_groups(lr=0.1, training_params={})
        updated_param_groups = kd_module.update_param_groups(param_groups=initial_param_groups, lr=0.2,
                                                             epoch=0, iter=0, training_params={},
                                                             total_batch=None)

        self.assertTrue(initial_param_groups[0]['lr'] == 0.2 == updated_param_groups[0]['lr'])

    def test_train_kd_module_external_models(self):
        sg_model = KDModel("test_train_kd_module_external_models", device='cpu')
        teacher_model = ResNet50(arch_params={}, num_classes=5)
        student_model = ResNet18(arch_params={}, num_classes=5)
        sg_model.connect_dataset_interface(self.dataset)

        sg_model.train(student=deepcopy(student_model),
                       teacher=teacher_model,
                       training_params=self.kd_train_params)

        # TEACHER WEIGHT'S SHOULD REMAIN THE SAME
        self.assertTrue(
            check_models_have_same_weights(teacher_model, sg_model.net.module.teacher))

        # STUDENT WEIGHT'S SHOULD NOT REMAIN THE SAME
        self.assertFalse(
            check_models_have_same_weights(student_model, sg_model.net.module.student))

    def test_train_model_with_input_adapter(self):
        kd_model = KDModel("train_kd_module_with_with_input_adapter", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        student = models.get('resnet18', arch_params={'num_classes': 5})
        teacher = models.get('resnet50', arch_params={'num_classes': 5},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"})
        kd_arch_params = {
            "teacher_input_adapter": NormalizationAdapter(mean_original=[0.485, 0.456, 0.406],
                                                          std_original=[0.229, 0.224, 0.225],
                                                          mean_required=[0.5, 0.5, 0.5],
                                                          std_required=[0.5, 0.5, 0.5])}
        kd_model.train(student=student,
                       teacher=teacher,
                       training_params=self.kd_train_params,
                       kd_arch_params=kd_arch_params)

        self.assertTrue(isinstance(kd_model.net.module.teacher[0], NormalizationAdapter))

    def test_load_ckpt_best_for_student(self):
        kd_model = KDModel("test_load_ckpt_best", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        student = models.get('resnet18', arch_params={'num_classes': 5})
        teacher = models.get('resnet50', arch_params={'num_classes': 5},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"})
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        kd_model.train(student=student, teacher=teacher, training_params=train_params)
        best_student_ckpt = os.path.join(kd_model.checkpoints_dir_path, "ckpt_best.pth")

        student_reloaded = models.get('resnet18', arch_params={'num_classes': 5},
                                      checkpoint_params={"checkpoint_path": best_student_ckpt})

        self.assertTrue(
            check_models_have_same_weights(student_reloaded, kd_model.net.module.student))

    def test_load_ckpt_best_for_student_with_ema(self):
        kd_model = KDModel("test_load_ckpt_best", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        student = models.get('resnet18', arch_params={'num_classes': 5})
        teacher = models.get('resnet50', arch_params={'num_classes': 5},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"})
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        train_params["ema"] = True
        kd_model.train(student=student, teacher=teacher, training_params=train_params)
        best_student_ckpt = os.path.join(kd_model.checkpoints_dir_path, "ckpt_best.pth")

        student_reloaded = models.get('resnet18', arch_params={'num_classes': 5},
                                      checkpoint_params={"checkpoint_path": best_student_ckpt})

        self.assertTrue(
            check_models_have_same_weights(student_reloaded, kd_model.ema_model.ema.module.student))

    def test_resume_kd_training(self):
        kd_model = KDModel("test_resume_training_start", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        student = models.get('resnet18', arch_params={'num_classes': 5})
        teacher = models.get('resnet50', arch_params={'num_classes': 5},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"})
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        kd_model.train(student=student, teacher=teacher, training_params=train_params)
        latest_net = deepcopy(kd_model.net)

        kd_model = KDModel("test_resume_training_start", device='cpu')
        kd_model.connect_dataset_interface(self.dataset)
        student = models.get('resnet18', arch_params={'num_classes': 5})
        teacher = models.get('resnet50', arch_params={'num_classes': 5},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"})

        train_params["max_epochs"] = 2
        train_params["resume"] = True
        collector = PreTrainingNetCollector()
        train_params["phase_callbacks"] = [collector]
        kd_model.train(student=student, teacher=teacher, training_params=train_params)

        self.assertTrue(
            check_models_have_same_weights(collector.net, latest_net))


if __name__ == '__main__':
    unittest.main()
