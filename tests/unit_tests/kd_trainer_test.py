import unittest
import os
from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.kd_model.kd_model import KDTrainer
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


class KDTrainerTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sg_trained_teacher = Trainer("sg_trained_teacher", device='cpu')
        cls.dataset_params = {"batch_size": 5}
        cls.dataset = ClassificationTestDatasetInterface(dataset_params=cls.dataset_params)
        cls.sg_trained_teacher.connect_dataset_interface(cls.dataset)

        cls.sg_trained_teacher.build_model('resnet50', arch_params={'num_classes': 5})

        cls.train_params = {"max_epochs": 3, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                            "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": torch.nn.CrossEntropyLoss(),
                            "optimizer": "SGD",
                            "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                            "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                            "metric_to_watch": "Accuracy",
                            "greater_metric_to_watch_is_better": True, "average_best_models": False}

        cls.kd_train_params = {"max_epochs": 3, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                               "lr_warmup_epochs": 0, "initial_lr": 0.1,
                               "loss": KDLogitsLoss(torch.nn.CrossEntropyLoss()),
                               "optimizer": "SGD",
                               "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                               "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                               "metric_to_watch": "Accuracy",
                               'loss_logging_items_names': ["Loss", "Task Loss", "Distillation Loss"],
                               "greater_metric_to_watch_is_better": True, "average_best_models": False}

    def test_build_kd_module_with_pretrained_teacher(self):
        kd_model = KDTrainer("build_kd_module_with_pretrained_teacher", device='cpu')
        kd_model.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                             )
        imagenet_resnet50_trainer = Trainer("pretrained_resnet50")
        imagenet_resnet50_trainer.build_model('resnet50', arch_params={'num_classes': 1000},
                                               checkpoint_params={'pretrained_weights': "imagenet"})

        self.assertTrue(check_models_have_same_weights(kd_model.net.module.teacher,
                                                       imagenet_resnet50_trainer.net.module))

    def test_build_kd_module_with_pretrained_student(self):
        kd_trainer = KDTrainer("build_kd_module_with_pretrained_student", device='cpu')
        kd_trainer.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'student_pretrained_weights': "imagenet",
                                                'teacher_pretrained_weights': "imagenet"}
                             )

        imagenet_resnet18_trainer = Trainer("pretrained_resnet18", device='cpu')
        imagenet_resnet18_trainer.build_model('resnet18', arch_params={'num_classes': 1000},
                                               checkpoint_params={'pretrained_weights': "imagenet"})

        self.assertTrue(check_models_have_same_weights(kd_trainer.net.module.student,
                                                       imagenet_resnet18_trainer.net.module))

    def test_build_kd_module_pretrained_student_with_head_replacement(self):
        self.sg_trained_teacher.train(self.train_params)
        teacher_path = os.path.join(self.sg_trained_teacher.checkpoints_dir_path, 'ckpt_latest.pth')

        kd_trainer = KDTrainer('test_build_kd_module_student_replace_head', device='cpu')
        kd_trainer.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 5}, teacher_arch_params={'num_classes': 5},
                                checkpoint_params={'student_pretrained_weights': "imagenet",
                                                   "teacher_checkpoint_path": teacher_path}
                                )

        self.assertTrue(kd_trainer.net.module.student.linear.out_features == 5)

    def test_build_kd_module_with_sg_trained_teacher(self):
        self.sg_trained_teacher.train(self.train_params)
        teacher_path = os.path.join(self.sg_trained_teacher.checkpoints_dir_path, 'ckpt_latest.pth')

        kd_trainer = KDTrainer('test_build_kd_module_with_sg_trained_teacher', device='cpu')

        kd_trainer.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 5}, teacher_arch_params={'num_classes': 5},
                                checkpoint_params={"teacher_checkpoint_path": teacher_path}
                                )

        self.assertTrue(
            check_models_have_same_weights(self.sg_trained_teacher.net.module, kd_trainer.net.module.teacher))

    def test_teacher_sg_module_methods(self):
        kd_trainer = KDTrainer("test_teacher_sg_module_methods", device='cpu')
        kd_trainer.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                                checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                                )

        initial_param_groups = kd_trainer.net.module.initialize_param_groups(lr=0.1, training_params={})
        updated_param_groups = kd_trainer.net.module.update_param_groups(param_groups=initial_param_groups, lr=0.2,
                                                                          epoch=0, iter=0, training_params={},
                                                                          total_batch=None)

        self.assertTrue(initial_param_groups[0]['lr'] == 0.2 == updated_param_groups[0]['lr'])

    def test_kd_architecture_kwarg_exception_catching(self):
        kd_trainer = KDTrainer("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(ArchitectureKwargsException):
            kd_trainer.build_model(teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 5}, teacher_arch_params={'num_classes': 5},
                                    checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                                    )

    def test_kd_unsupported_kdmodel_arg_exception_catching(self):
        kd_trainer = KDTrainer("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(UnsupportedKDModelArgException):
            kd_trainer.build_model(student_architecture='resnet18',
                                    teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 1000},
                                    teacher_arch_params={'num_classes': 1000},
                                    checkpoint_params={"pretrained_weights": "imagenet"},
                                    )

    def test_kd_unsupported_model_exception_catching(self):
        kd_trainer = KDTrainer("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(UnsupportedKDArchitectureException):
            kd_trainer.build_model(student_architecture='resnet18',
                                    teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 1000},
                                    teacher_arch_params={'num_classes': 1000},
                                    checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                                    architecture='unsupported_model'
                                    )

    def test_kd_inconsistent_params_exception_catching(self):
        kd_trainer = KDTrainer("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(InconsistentParamsException):
            kd_trainer.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 10}, teacher_arch_params={'num_classes': 1000},
                                    checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                                    )

    def test_kd_teacher_knowledge_exception_catching(self):
        kd_trainer = KDTrainer("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(TeacherKnowledgeException):
            kd_trainer.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000}
                                    )

    def test_build_external_models(self):
        kd_trainer = KDTrainer("test_training_with_external_teacher", device='cpu')
        teacher_model = ResNet50(arch_params={}, num_classes=10)
        student_model = ResNet18(arch_params={}, num_classes=10)
        kd_trainer.build_model(student_architecture=student_model, teacher_architecture=teacher_model,
                             student_arch_params={'num_classes': 10}, teacher_arch_params={'num_classes': 10}
                             )

        self.assertTrue(
            check_models_have_same_weights(teacher_model, kd_trainer.net.module.teacher))
        self.assertTrue(
            check_models_have_same_weights(student_model, kd_trainer.net.module.student))

    def test_train_kd_module_external_models(self):
        kd_trainer = KDTrainer("test_train_kd_module_external_models", device='cpu')
        teacher_model = ResNet50(arch_params={}, num_classes=5)
        student_model = ResNet18(arch_params={}, num_classes=5)
        kd_trainer.connect_dataset_interface(self.dataset)
        kd_trainer.build_model(run_teacher_on_eval=True,
                             student_arch_params={'num_classes': 5},
                             teacher_arch_params={'num_classes': 5},
                             student_architecture=deepcopy(student_model),
                             teacher_architecture=deepcopy(teacher_model),
                             )

        kd_trainer.train(self.kd_train_params)

        # TEACHER WEIGHT'S SHOULD REMAIN THE SAME
        self.assertTrue(
            check_models_have_same_weights(teacher_model, kd_trainer.net.module.teacher))

        # STUDENT WEIGHT'S SHOULD NOT REMAIN THE SAME
        self.assertFalse(
            check_models_have_same_weights(student_model, kd_trainer.net.module.student))

    def test_train_kd_module_pretrained_ckpt(self):
        kd_trainer = KDTrainer("test_train_kd_module_pretrained_ckpt", device='cpu')
        teacher_model = ResNet50(arch_params={}, num_classes=5)
        teacher_path = '/tmp/teacher.pth'
        torch.save(teacher_model.state_dict(), teacher_path)
        kd_trainer.connect_dataset_interface(self.dataset)

        kd_trainer.build_model(student_arch_params={'num_classes': 5},
                             teacher_arch_params={'num_classes': 5},
                             student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             checkpoint_params={"teacher_checkpoint_path": teacher_path}
                             )
        kd_trainer.train(self.kd_train_params)

    def test_build_model_with_input_adapter(self):
        kd_model = KDTrainer("build_kd_module_with_with_input_adapter", device='cpu')
        kd_model.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"},
                             arch_params={
                                 "teacher_input_adapter": NormalizationAdapter(mean_original=[0.485, 0.456, 0.406],
                                                                               std_original=[0.229, 0.224, 0.225],
                                                                               mean_required=[0.5, 0.5, 0.5],
                                                                               std_required=[0.5, 0.5, 0.5])})
        self.assertTrue(isinstance(kd_model.net.module.teacher[0], NormalizationAdapter))

    def test_load_ckpt_best_for_student(self):
        kd_trainer = KDTrainer("test_load_ckpt_best", device='cpu')
        teacher_model = ResNet50(arch_params={}, num_classes=5)
        teacher_path = '/tmp/teacher.pth'
        torch.save(teacher_model.state_dict(), teacher_path)
        kd_trainer.connect_dataset_interface(self.dataset)

        kd_trainer.build_model(student_arch_params={'num_classes': 5},
                             teacher_arch_params={'num_classes': 5},
                             student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             checkpoint_params={"teacher_checkpoint_path": teacher_path}
                             )
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        kd_trainer.train(train_params)
        best_student_ckpt = os.path.join(kd_trainer.checkpoints_dir_path, "ckpt_best.pth")

        student_sg_model = Trainer("studnet_sg_model")
        student_sg_model.build_model("resnet18", arch_params={'num_classes': 5},
                                     checkpoint_params={"load_checkpoint": True, "external_checkpoint_path": best_student_ckpt})

        self.assertTrue(
            check_models_have_same_weights(student_sg_model.net.module, kd_trainer.net.module.student))

    def test_load_ckpt_best_for_student_with_ema(self):
        kd_trainer = KDTrainer("test_load_ckpt_best_for_student_with_ema", device='cpu')
        teacher_model = ResNet50(arch_params={}, num_classes=5)
        teacher_path = '/tmp/teacher.pth'
        torch.save(teacher_model.state_dict(), teacher_path)
        kd_trainer.connect_dataset_interface(self.dataset)

        kd_trainer.build_model(student_arch_params={'num_classes': 5},
                             teacher_arch_params={'num_classes': 5},
                             student_architecture='resnet18',
                             teacher_architecture='resnet50',
                             checkpoint_params={"teacher_checkpoint_path": teacher_path}
                             )
        train_params = self.kd_train_params.copy()
        train_params["max_epochs"] = 1
        train_params["ema"] = True
        kd_trainer.train(train_params)
        best_student_ckpt = os.path.join(kd_trainer.checkpoints_dir_path, "ckpt_best.pth")

        student_trainer = Trainer("studnet_sg_model")
        student_trainer.build_model("resnet18", arch_params={'num_classes': 5},
                                     checkpoint_params={"load_checkpoint": True, "external_checkpoint_path": best_student_ckpt})
        self.assertTrue(
            check_models_have_same_weights(student_trainer.net.module, kd_trainer.ema_model.ema.module.student))


if __name__ == '__main__':
    unittest.main()
