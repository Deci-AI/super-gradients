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
    TeacherKnowledgeException, UndefinedNumClassesException


class KDModuleTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.sg_trained_teacher = SgModel("sg_trained_teacher", device='cpu')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        cls.sg_trained_teacher.connect_dataset_interface(dataset)

        cls.sg_trained_teacher.build_model('resnet50', arch_params={'num_classes': 5})

        cls.train_params = {"max_epochs": 3, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                            "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": torch.nn.CrossEntropyLoss(),
                            "optimizer": "SGD",
                            "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                            "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                            "metric_to_watch": "Accuracy",
                            "greater_metric_to_watch_is_better": True}

    def test_build_kd_module_with_pretrained_teacher(self):
        kd_model = KDModel("build_kd_module_with_pretrained_teacher", device='cpu')
        kd_model.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                             )
        imagenet_resnet50_sg_model = SgModel("pretrained_resnet50")
        imagenet_resnet50_sg_model.build_model('resnet50', arch_params={'num_classes': 1000},
                                               checkpoint_params={'pretrained_weights': "imagenet"})

        self.assertTrue(check_models_have_same_weights(kd_model.net.module.teacher,
                                                       imagenet_resnet50_sg_model.net.module))

    def test_build_kd_module_with_pretrained_student(self):
        kd_model = KDModel("build_kd_module_with_pretrained_student", device='cpu')
        kd_model.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                             student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                             checkpoint_params={'student_pretrained_weights': "imagenet",
                                                'teacher_pretrained_weights': "imagenet"}
                             )

        imagenet_resnet18_sg_model = SgModel("pretrained_resnet18", device='cpu')
        imagenet_resnet18_sg_model.build_model('resnet18', arch_params={'num_classes': 1000},
                                               checkpoint_params={'pretrained_weights': "imagenet"})

        self.assertTrue(check_models_have_same_weights(kd_model.net.module.student,
                                                       imagenet_resnet18_sg_model.net.module))

    def test_build_kd_module_pretrained_student_with_head_replacement(self):
        self.sg_trained_teacher.train(self.train_params)
        teacher_path = os.path.join(self.sg_trained_teacher.checkpoints_dir_path, 'ckpt_latest.pth')

        sg_kd_model = KDModel('test_build_kd_module_student_replace_head', device='cpu')
        sg_kd_model.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 5}, teacher_arch_params={'num_classes': 5},
                                checkpoint_params={'student_pretrained_weights': "imagenet",
                                                   "teacher_checkpoint_path": teacher_path}
                                )

        self.assertTrue(sg_kd_model.net.module.student.linear.out_features == 5)

    def test_build_kd_module_with_sg_trained_teacher(self):
        self.sg_trained_teacher.train(self.train_params)
        teacher_path = os.path.join(self.sg_trained_teacher.checkpoints_dir_path, 'ckpt_latest.pth')

        sg_kd_model = KDModel('test_build_kd_module_with_sg_trained_teacher', device='cpu')

        sg_kd_model.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 5}, teacher_arch_params={'num_classes': 5},
                                checkpoint_params={"teacher_checkpoint_path": teacher_path}
                                )

        self.assertTrue(
            check_models_have_same_weights(self.sg_trained_teacher.net.module, sg_kd_model.net.module.teacher))

    def test_teacher_sg_module_methods(self):
        sg_kd_model = KDModel("test_teacher_sg_module_methods", device='cpu')
        sg_kd_model.build_model(student_architecture='resnet18', teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                                checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                                )

        initial_param_groups = sg_kd_model.net.module.initialize_param_groups(lr=0.1, training_params={})
        updated_param_groups = sg_kd_model.net.module.update_param_groups(param_groups=initial_param_groups, lr=0.2,
                                                                          epoch=0, iter=0, training_params={},
                                                                          total_batch=None)

        self.assertTrue(initial_param_groups[0]['lr'] == 0.2 == updated_param_groups[0]['lr'])

    def test_kd_architecture_kwarg_sexception_catching(self):
        sg_kd_model = KDModel("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(ArchitectureKwargsException):
            sg_kd_model.build_model(teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 5}, teacher_arch_params={'num_classes': 5},
                                    checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                                    )

    def test_kd_unsupported_kdmodel_arg_exceptione_catching(self):
        sg_kd_model = KDModel("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(UnsupportedKDModelArgException):
            sg_kd_model.build_model(student_architecture='resnet18',
                                    teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 1000},
                                    teacher_arch_params={'num_classes': 1000},
                                    checkpoint_params={"pretrained_weights": "imagenet"},
                                    )

    def test_kd_unsupported_model_exception_catching(self):
        sg_kd_model = KDModel("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(UnsupportedKDArchitectureException):
            sg_kd_model.build_model(teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000},
                                    checkpoint_params={'teacher_pretrained_weights': "imagenet"}, architecture='unsupported_model'
                                    )

    def test_kd_inconsistent_params_exception_catching(self):
        sg_kd_model = KDModel("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(InconsistentParamsException):
            sg_kd_model.build_model(student_architecture='resnet18',teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 10}, teacher_arch_params={'num_classes': 1000},
                                    checkpoint_params={'teacher_pretrained_weights': "imagenet"}
                                    )

    def test_kd_teacher_knowledge_exception_catching(self):
        sg_kd_model = KDModel("test_teacher_sg_module_methods", device='cpu')
        with self.assertRaises(TeacherKnowledgeException):
            sg_kd_model.build_model(student_architecture='resnet18',teacher_architecture='resnet50',
                                    student_arch_params={'num_classes': 1000}, teacher_arch_params={'num_classes': 1000}
                                    )

if __name__ == '__main__':
    unittest.main()
