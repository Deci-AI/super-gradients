import unittest
import super_gradients
from super_gradients.training import MultiGPUMode
from super_gradients.training import Trainer
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy
import os
import shutil


class PretrainedModelsUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.imagenet_pretrained_models = ["resnet50", "repvgg_a0", "regnetY800"]

        self.test_dataset = ClassificationTestDatasetInterface(classes=range(1000))

    def test_pretrained_resnet50_imagenet(self):
        trainer = Trainer('imagenet_pretrained_resnet50_unit_test', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.test_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet50", checkpoint_params={"pretrained_weights": "imagenet"})
        trainer.test(test_loader=self.test_dataset.val_loader, test_metrics_list=[Accuracy()],
                     metrics_progress_verbose=True)

    def test_pretrained_regnetY800_imagenet(self):
        trainer = Trainer('imagenet_pretrained_regnetY800_unit_test', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.test_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY800", checkpoint_params={"pretrained_weights": "imagenet"})
        trainer.test(test_loader=self.test_dataset.val_loader, test_metrics_list=[Accuracy()],
                     metrics_progress_verbose=True)

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = Trainer('imagenet_pretrained_repvgg_a0_unit_test', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.test_dataset, data_loader_num_workers=8)
        trainer.build_model("repvgg_a0", checkpoint_params={"pretrained_weights": "imagenet"},
                            arch_params={"build_residual_branches": True})
        trainer.test(test_loader=self.test_dataset.val_loader, test_metrics_list=[Accuracy()],
                     metrics_progress_verbose=True)

    def tearDown(self) -> None:
        if os.path.exists('~/.cache/torch/hub/'):
            shutil.rmtree('~/.cache/torch/hub/')


if __name__ == '__main__':
    unittest.main()
