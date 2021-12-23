import unittest
import super_gradients
from super_gradients.training import MultiGPUMode
from super_gradients.training import SgModel
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface, \
    ClassificationTestDatasetInterface, CityscapesDatasetInterface, SegmentationTestDatasetInterface
from super_gradients.training.metrics import Accuracy, IoU
import os
import shutil
import torchvision.transforms as transforms
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss


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

        self.transfer_classification_dataset = ClassificationTestDatasetInterface(image_size=224)

        self.transfer_classification_train_params = {"max_epochs": 3,
                                                     "lr_updates": [1],
                                                     "lr_decay_factor": 0.1,
                                                     "initial_lr": 0.6,
                                                     "loss": "cross_entropy",
                                                     "lr_mode": "step",
                                                     "optimizer_params": {"weight_decay": 0.000,
                                                                          "momentum": 0.9},
                                                     "train_metrics_list": [Accuracy()],
                                                     "valid_metrics_list": [Accuracy()],
                                                     "loss_logging_items_names": ["Loss"],
                                                     "metric_to_watch": "Accuracy",
                                                     "greater_metric_to_watch_is_better": True}

        self.cityscapes_pretrained_models = ["ddrnet_23", "ddrnet_23_slim"]
        self.cityscapes_pretrained_arch_params = {
            "ddrnet_23": {"pretrained_weights": "cityscapes", "num_classes": 19, "aux_head": True, "sync_bn": True}}
        self.cityscapes_pretrained_mious = {"ddrnet_23": 0.7865,
                                            "ddrnet_23_slim": 0.7689}
        self.cityscapes_dataset = CityscapesDatasetInterface(dataset_params={
            "batch_size": 3,
            "val_batch_size": 3,
            "dataset_dir": "/home/ofri/cityscapes/",
            "crop_size": 1024,
            "img_size": 1024,
            "image_mask_transforms_aug": transforms.Compose([]),
            "image_mask_transforms": transforms.Compose([])  # no transform for evaluation
        }, cache_labels=False)
        self.transfer_segmentation_dataset = SegmentationTestDatasetInterface(image_size=1024)
        self.transfer_segmentation_train_params = {"max_epochs": 3,
                                                   "initial_lr": 1e-2,
                                                   "loss": DDRNetLoss(),
                                                   "lr_mode": "poly",
                                                   "ema": True,  # unlike the paper (not specified in paper)
                                                   "average_best_models": True,
                                                   "optimizer": "SGD",
                                                   "mixed_precision": False,
                                                   "optimizer_params":
                                                       {"weight_decay": 5e-4,
                                                        "momentum": 0.9},
                                                   "load_opt_params": False,
                                                   "train_metrics_list": [IoU(5)],
                                                   "valid_metrics_list": [IoU(5)],
                                                   "loss_logging_items_names": ["main_loss", "aux_loss", "Loss"],
                                                   "metric_to_watch": "IoU",
                                                   "greater_metric_to_watch_is_better": True
                                                   }

    def test_pretrained_resnet50_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet50', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet50", arch_params=self.imagenet_pretrained_arch_params["resnet50"])
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet50"])

    def test_transfer_learning_resnet50_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet50_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet50", arch_params=self.imagenet_pretrained_arch_params["resnet50"])
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_regnetY800_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY800', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY800", arch_params=self.imagenet_pretrained_arch_params["regnetY800"])
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY800"])

    def test_transfer_learning_regnetY800_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY800_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY800", arch_params=self.imagenet_pretrained_arch_params["regnetY800"])
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = SgModel('imagenet_pretrained_repvgg_a0', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("repvgg_a0", arch_params=self.imagenet_pretrained_arch_params["repvgg_a0"])
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["repvgg_a0"])

    def test_transfer_learning_repvgg_a0_imagenet(self):
        trainer = SgModel('imagenet_pretrained_repvgg_a0_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("repvgg_a0", arch_params=self.imagenet_pretrained_arch_params["repvgg_a0"])
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_ddrnet23_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"])
        res = trainer.test(test_loader=self.cityscapes_dataset.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["ddrnet_23"])

    def test_pretrained_ddrnet23_slim_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23_slim', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23_slim", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"])
        res = trainer.test(test_loader=self.cityscapes_dataset.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["ddrnet_23_slim"])

    def test_transfer_learning_ddrnet23_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"])
        trainer.train(training_params=self.transfer_segmentation_train_params)

    def test_transfer_learning_ddrnet23_slim_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23_slim_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23_slim", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"])
        trainer.train(training_params=self.transfer_segmentation_train_params)

    def tearDown(self) -> None:
        if os.path.exists('~/.cache/torch/hub/'):
            shutil.rmtree('~/.cache/torch/hub/')


if __name__ == '__main__':
    unittest.main()
