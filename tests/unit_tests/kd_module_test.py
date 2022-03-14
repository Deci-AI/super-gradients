import unittest
import os
from super_gradients.training import SgModel
import torch
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy


class KDModuleTest(unittest.TestCase):
    def test_build_kd_module_with_pretrained_teacher(self):
        sg_model = SgModel("build_kd_module_with_pretrained_teacher", device='cpu')
        sg_model.build_kd_model(student_architecture='resnet18',
                                teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 1000},
                                teacher_arch_params={'pretrained_weights': "imagenet"},
                                )
        imagenet_resnet50_sg_model = SgModel("pretrained_resnet50")
        imagenet_resnet50_sg_model.build_model('resnet50', arch_params={'pretrained_weights': "imagenet",
                                                                        'num_classes': 1000})

        self.assertTrue(check_models_have_same_weights(sg_model.net.module.teacher,
                                                       imagenet_resnet50_sg_model.net.module))

    def test_build_kd_module_with_sg_trained_teacher(self):
        sg_model = SgModel("sg_trained_teacher", device='cpu')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        sg_model.connect_dataset_interface(dataset)

        sg_model.build_model('resnet50', arch_params={'num_classes': 5})

        train_params = {"max_epochs": 3, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": torch.nn.CrossEntropyLoss(),
                        "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True}
        sg_model.train(train_params)

        teacher_path = os.path.join(sg_model.checkpoints_dir_path, 'ckpt_latest.pth')

        sg_kd_model = SgModel('test_build_kd_module_with_sg_trained_teacher', device='cpu')
        sg_kd_model.build_kd_model(student_architecture='resnet18',
                                   teacher_architecture='resnet50',
                                   student_arch_params={'num_classes': 5},
                                   teacher_arch_params={'num_classes': 5},
                                   teacher_checkpoint_path=teacher_path
                                   )

        self.assertTrue(check_models_have_same_weights(sg_model.net.module, sg_kd_model.net.module.teacher))

    def test_build_kd_module_with_pretrained_teacher(self):
        sg_model = SgModel("test_teacher_sg_module_methods", device='cpu')
        sg_model.build_kd_model(student_architecture='resnet18',
                                teacher_architecture='resnet50',
                                student_arch_params={'num_classes': 1000},
                                teacher_arch_params={'pretrained_weights': "imagenet"},
                                )

        initial_param_groups = sg_model.net.module.initialize_param_groups(lr=0.1, training_params={})
        updated_param_groups = sg_model.net.module.update_param_groups(param_groups=initial_param_groups, lr=0.2,
                                                                       epoch=0, iter=0, training_params={},
                                                                       total_batch=None)

        self.assertTrue(initial_param_groups[0]['lr'] == 0.2 == updated_param_groups[0]['lr'])


if __name__ == '__main__':
    unittest.main()
