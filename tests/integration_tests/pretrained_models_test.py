import unittest
import super_gradients
from super_gradients.training import MultiGPUMode
from super_gradients.training import SgModel
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface, \
    CoCoDetectionDatasetInterface, DetectionTestDatasetInterface,     ClassificationTestDatasetInterface, CityscapesDatasetInterface, SegmentationTestDatasetInterface, \
    CoCoSegmentationDatasetInterface
from super_gradients.training.metrics import Accuracy, DetectionMetrics, IoU
import os
import shutil
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.models.detection_models.yolov5 import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import Anchors
import torchvision.transforms as transforms
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss
from super_gradients.training.utils.segmentation_utils import coco_sub_classes_inclusion_tuples_list

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
        self.coco_pretrained_models = ["yolo_v5s", "yolo_v5m"]
        self.coco_pretrained_arch_params = {"yolo_v5": {"pretrained_weights": "coco"}}
        self.coco_dataset = CoCoDetectionDatasetInterface(dataset_params={"batch_size": 64,
                                                                          "val_batch_size": 64,
                                                                          "train_image_size": 640,
                                                                          "val_image_size": 640,
                                                                          "val_collate_fn": base_detection_collate_fn,
                                                                          "val_collate_fn": base_detection_collate_fn,
                                                                          "val_sample_loading_method": "rectangular",
                                                                          "dataset_hyper_param": {
                                                                              "hsv_h": 0.015,
                                                                              "hsv_s": 0.7,
                                                                              "hsv_v": 0.4,
                                                                              "degrees": 0.0,
                                                                              "translate": 0.1,
                                                                              "scale": 0.5,  # IMAGE SCALE (+/- gain)
                                                                              "shear": 0.0}  # IMAGE SHEAR (+/- deg)
                                                                          })
        self.coco_pretrained_maps = {"yolo_v5s": 36.73585628224802}
        self.transfer_detection_dataset = DetectionTestDatasetInterface(image_size=640)
        self.transfer_detection_train_params = {"max_epochs": 3,
                                                "lr_mode": "cosine",
                                                "initial_lr": 0.01,
                                                "cosine_final_lr_ratio": 0.2,
                                                "lr_warmup_epochs": 3,
                                                "batch_accumulate": 1,
                                                "warmup_bias_lr": 0.1,
                                                "loss": "yolo_v5_loss",
                                                "criterion_params": {"anchors": Anchors(anchors_list=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
                                                                                        strides=[8, 16, 32]),
                                                                     "obj_loss_gain": 1.0,
                                                                     "box_loss_gain": 0.05,
                                                                     "cls_loss_gain": 0.5,
                                                                     },
                                                "optimizer": "SGD",
                                                "warmup_momentum": 0.8,
                                                "optimizer_params": {"momentum": 0.937,
                                                                     "weight_decay": 0.0005,
                                                                     "nesterov": True},
                                                "train_metrics_list": [],
                                                "valid_metrics_list": [
                                                    DetectionMetrics(post_prediction_callback=YoloV5PostPredictionCallback(),
                                                                     num_cls=len(
                                                                   self.coco_dataset.coco_classes))],
                                                "loss_logging_items_names": ["GIoU", "obj", "cls", "Loss"],
                                                "metric_to_watch": "mAP@0.50:0.95",
                                                "greater_metric_to_watch_is_better": True}




        self.coco_segmentation_subclass_pretrained_models = ["shelfnet34_lw"]
        self.coco_segmentation_subclass_pretrained_arch_params = {
            "shelfnet34_lw": {"pretrained_weights": "coco_segmentation_subclass",
                           "num_classes": 21, "image_size": 512}}
        self.coco_segmentation_subclass_pretrained_mious = {"shelfnet34_lw": 0.651}
        self.coco_segmentation_dataset = CoCoSegmentationDatasetInterface(dataset_params={
            "batch_size": 24,
            "val_batch_size": 24,
            "dataset_dir": "/data/coco/",
            "img_size": 608,
            "crop_size": 512
        }, dataset_classes_inclusion_tuples_list=coco_sub_classes_inclusion_tuples_list()
        )

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

    def test_pretrained_coco_segmentation_subclass_pretrained_shelfnet34_lw(self):
        trainer = SgModel('coco_segmentation_subclass_pretrained_shelfnet34_lw', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.coco_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("shelfnet34_lw",
                            arch_params=self.coco_segmentation_subclass_pretrained_arch_params["shelfnet34_lw"])
        res = trainer.test(test_loader=self.coco_segmentation_dataset.val_loader, test_metrics_list=[IoU(21)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.coco_segmentation_subclass_pretrained_mious["shelfnet34_lw"], places=2)

    def test_pretrained_yolov5s_coco(self):
        trainer = SgModel('coco_pretrained_yolov5s', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.coco_dataset, data_loader_num_workers=8)
        trainer.build_model("yolo_v5s", arch_params=self.coco_pretrained_arch_params["yolo_v5"])
        res = trainer.test(test_loader=self.coco_dataset.val_loader,
                           test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloV5PostPredictionCallback(),
                                                               num_cls=len(
                                                                   self.coco_dataset.coco_classes))],
                           metrics_progress_verbose=True)[3]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolo_v5s"])

    def test_transfer_learning_yolov5s_coco(self):
        trainer = SgModel('coco_pretrained_yolov5s', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_detection_dataset, data_loader_num_workers=8)
        trainer.build_model("yolo_v5s", arch_params=self.coco_pretrained_arch_params["yolo_v5"])
        trainer.train(training_params=self.transfer_detection_train_params)

    def tearDown(self) -> None:
        if os.path.exists('~/.cache/torch/hub/'):
            shutil.rmtree('~/.cache/torch/hub/')


if __name__ == '__main__':
    unittest.main()
