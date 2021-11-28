import unittest
import super_gradients
from super_gradients.training import MultiGPUMode
from super_gradients.training import SgModel
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface
from super_gradients.training.metrics import Accuracy
import os
import shutil


class PretrainedModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.imagenet_pretrained_models = ["resnet50", "repvgg_a0", "regnetY800"]

        self.imagenet_pretrained_arch_params = {"resnet50": {"pretrained_weights": "imagenet"},
                                                "regnetY800": {"pretrained_weights": "imagenet"},
                                                "repvgg_a0": {"pretrained_weights": "imagenet",
                                                              "build_residual_branches": True}}

        self.imagenet_pretrained_accuracies = {"resnet50": 0.763,
                                               "repvgg_a0": 0.7205,
                                               "regnetY800": 0.7605}
        self.imagenet_dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params={"batch_size": 128})

    def test_pretrained_resnet50_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet50', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet50", arch_params=self.imagenet_pretrained_arch_params["resnet50"])
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet50"])

    def test_pretrained_regnetY800_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY800', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY800", arch_params=self.imagenet_pretrained_arch_params["regnetY800"])
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet50"])

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = SgModel('imagenet_pretrained_repvgg_a0', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("repvgg_a0", arch_params=self.imagenet_pretrained_arch_params["repvgg_a0"])
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet50"])

    def tearDown(self) -> None:
        if os.path.exists('~/.cache/torch/hub/'):
            shutil.rmtree('~/.cache/torch/hub/')


if __name__ == '__main__':
    unittest.main()
