import unittest
import super_gradients
from super_gradients.training import MultiGPUMode
from super_gradients.training import SgModel
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ImageNetDatasetInterface, \
    ClassificationTestDatasetInterface, CityscapesDatasetInterface, SegmentationTestDatasetInterface, \
    CoCoSegmentationDatasetInterface, CoCoDetectionDatasetInterface, DetectionTestDatasetInterface
from super_gradients.training.utils.segmentation_utils import coco_sub_classes_inclusion_tuples_list
from super_gradients.training.metrics import Accuracy, IoU
import os
import shutil
from super_gradients.training.models.detection_models.yolov5 import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import Anchors
import torchvision.transforms as transforms
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.transforms.transforms import Rescale
from super_gradients.training.losses.stdc_loss import STDCLoss


class PretrainedModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.imagenet_pretrained_arch_params = {"resnet": {},
                                                "regnet": {},
                                                "repvgg_a0": {"build_residual_branches": True},
                                                "efficientnet_b0": {},
                                                "mobilenet": {},
                                                "vit_base":
                                                    {"image_size": (224, 224),
                                                     "patch_size": (16, 16)}}

        self.imagenet_pretrained_ckpt_params = {"pretrained_weights": "imagenet"}

        self.imagenet21k_pretrained_ckpt_params = {"pretrained_weights": "imagenet21k"}

        self.imagenet_pretrained_accuracies = {"resnet50": 0.7947,
                                               "resnet34": 0.7413,
                                               "resnet18": 0.706,
                                               "repvgg_a0": 0.7205,
                                               "regnetY800": 0.7707,
                                               "regnetY600": 0.7618,
                                               "regnetY400": 0.7474,
                                               "regnetY200": 0.7088,
                                               "efficientnet_b0": 0.7762,
                                               "mobilenet_v3_large": 0.7452,
                                               "mobilenet_v3_small": 0.6745,
                                               "mobilenet_v2": 0.7308,
                                               "vit_base": 0.8415,
                                               "vit_large": 0.8564,
                                               "beit_base_patch16_224": 0.85
                                               }
        self.imagenet_dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params={"batch_size": 128})

        self.imagenet_dataset_05_mean_std = ImageNetDatasetInterface(data_dir="/data/Imagenet",
                                                                     dataset_params={"batch_size": 128,
                                                                                     "img_mean": [0.5, 0.5, 0.5],
                                                                                     "img_std": [0.5, 0.5, 0.5],
                                                                                     "resize_size": 248
                                                                                     })

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
        self.coco_pretrained_arch_params = {"yolo_v5": {}}
        self.coco_pretrained_ckpt_params = {"pretrained_weights": "coco"}
        self.coco_dataset = CoCoDetectionDatasetInterface(dataset_params={"batch_size": 8,
                                                                          "val_batch_size": 8,
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
        self.coco_pretrained_maps = {"yolo_v5s": 0.362, "yolo_v5m": 0.441, "yolo_v5l": 0.471, "yolo_v5n": 0.267}
        self.transfer_detection_dataset = DetectionTestDatasetInterface(image_size=640)
        self.transfer_detection_train_params = {"max_epochs": 3,
                                                "lr_mode": "cosine",
                                                "initial_lr": 0.01,
                                                "cosine_final_lr_ratio": 0.2,
                                                "lr_warmup_epochs": 3,
                                                "batch_accumulate": 1,
                                                "warmup_bias_lr": 0.1,
                                                "loss": "yolo_v5_loss",
                                                "criterion_params": {"anchors": Anchors(
                                                    anchors_list=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                                                                  [116, 90, 156, 198, 373, 326]],
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
                                                    DetectionMetrics(
                                                        post_prediction_callback=YoloV5PostPredictionCallback(),
                                                        num_cls=len(
                                                            self.coco_dataset.coco_classes))],
                                                "loss_logging_items_names": ["GIoU", "obj", "cls", "Loss"],
                                                "metric_to_watch": "mAP@0.50:0.95",
                                                "greater_metric_to_watch_is_better": True}

        self.coco_segmentation_subclass_pretrained_arch_params = {
            "shelfnet34_lw": {"num_classes": 21, "image_size": 512}}
        self.coco_segmentation_subclass_pretrained_ckpt_params = {"pretrained_weights": "coco_segmentation_subclass"}
        self.coco_segmentation_subclass_pretrained_mious = {"shelfnet34_lw": 0.651}
        self.coco_segmentation_dataset = CoCoSegmentationDatasetInterface(dataset_params={
            "batch_size": 24,
            "val_batch_size": 24,
            "dataset_dir": "/data/coco/",
            "img_size": 608,
            "crop_size": 512
        }, dataset_classes_inclusion_tuples_list=coco_sub_classes_inclusion_tuples_list()
        )

        self.cityscapes_pretrained_models = ["ddrnet_23", "ddrnet_23_slim", "stdc1_seg50", "regseg48"]
        self.cityscapes_pretrained_arch_params = {
            "ddrnet_23": {"aux_head": True, "sync_bn": True},
            "regseg48": {},
            "stdc": {"use_aux_heads": True, "aux_head": True}}

        self.cityscapes_pretrained_ckpt_params = {"pretrained_weights": "cityscapes"}
        self.cityscapes_pretrained_mious = {"ddrnet_23": 0.7865,
                                            "ddrnet_23_slim": 0.7689,
                                            "stdc1_seg50": 0.7436,
                                            "stdc1_seg75": 0.7687,
                                            "stdc2_seg50": 0.7527,
                                            "stdc2_seg75": 0.7893,
                                            "regseg48": 0.7815}

        self.cityscapes_dataset = CityscapesDatasetInterface(dataset_params={
            "batch_size": 3,
            "val_batch_size": 3,
            "dataset_dir": "/data/cityscapes/",
            "crop_size": 1024,
            "img_size": 1024,
            "image_mask_transforms_aug": transforms.Compose([]),
            "image_mask_transforms": transforms.Compose([])  # no transform for evaluation
        }, cache_labels=False)

        self.cityscapes_dataset_rescaled50 = CityscapesDatasetInterface(dataset_params={
            "batch_size": 3,
            "val_batch_size": 3,
            "image_mask_transforms_aug": transforms.Compose([]),
            "image_mask_transforms": transforms.Compose([Rescale(scale_factor=0.5)])  # no transform for evaluation
        }, cache_labels=False)

        self.cityscapes_dataset_rescaled75 = CityscapesDatasetInterface(dataset_params={
            "batch_size": 3,
            "val_batch_size": 3,
            "image_mask_transforms_aug": transforms.Compose([]),
            "image_mask_transforms": transforms.Compose([Rescale(scale_factor=0.75)])  # no transform for evaluation
        }, cache_labels=False)

        self.transfer_segmentation_dataset = SegmentationTestDatasetInterface(image_size=1024)
        self.ddrnet_transfer_segmentation_train_params = {"max_epochs": 3,
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

        self.stdc_transfer_segmentation_train_params = {"max_epochs": 3,
                                                        "initial_lr": 1e-2,
                                                        "loss": STDCLoss(num_classes=5),
                                                        "lr_mode": "poly",
                                                        "ema": True,  # unlike the paper (not specified in paper)
                                                        "optimizer": "SGD",
                                                        "optimizer_params":
                                                            {"weight_decay": 5e-4,
                                                             "momentum": 0.9},
                                                        "load_opt_params": False,
                                                        "train_metrics_list": [IoU(5)],
                                                        "valid_metrics_list": [IoU(5)],
                                                        "loss_logging_items_names": ["main_loss", "aux_loss1", "aux_loss2", "detail_loss", "loss"],
                                                        "metric_to_watch": "IoU",
                                                        "greater_metric_to_watch_is_better": True
                                                        }

        self.regseg_transfer_segmentation_train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": "cross_entropy",
            "lr_mode": "poly",
            "ema": True,  # unlike the paper (not specified in paper)
            "optimizer": "SGD",
            "optimizer_params":
                {
                    "weight_decay": 5e-4,
                    "momentum": 0.9
                },
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "loss_logging_items_names": ["loss"],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True
        }

    def test_pretrained_resnet50_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet50', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet50", arch_params=self.imagenet_pretrained_arch_params["resnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet50"], delta=0.001)

    def test_transfer_learning_resnet50_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet50_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet50", arch_params=self.imagenet_pretrained_arch_params["resnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_resnet34_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet34', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet34", arch_params=self.imagenet_pretrained_arch_params["resnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet34"], delta=0.001)

    def test_transfer_learning_resnet34_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet34_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet34", arch_params=self.imagenet_pretrained_arch_params["resnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_resnet18_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet18', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet18", arch_params=self.imagenet_pretrained_arch_params["resnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet18"], delta=0.001)

    def test_transfer_learning_resnet18_imagenet(self):
        trainer = SgModel('imagenet_pretrained_resnet18_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("resnet18", arch_params=self.imagenet_pretrained_arch_params["resnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_regnetY800_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY800', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY800", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY800"], delta=0.001)

    def test_transfer_learning_regnetY800_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY800_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY800", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_regnetY600_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY600', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY600", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY600"], delta=0.001)

    def test_transfer_learning_regnetY600_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY600_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY600", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_regnetY400_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY400', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY400", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY400"], delta=0.001)

    def test_transfer_learning_regnetY400_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY400_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY400", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_regnetY200_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY200', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY200", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY200"], delta=0.001)

    def test_transfer_learning_regnetY200_imagenet(self):
        trainer = SgModel('imagenet_pretrained_regnetY200_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("regnetY200", arch_params=self.imagenet_pretrained_arch_params["regnet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = SgModel('imagenet_pretrained_repvgg_a0', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("repvgg_a0", arch_params=self.imagenet_pretrained_arch_params["repvgg_a0"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["repvgg_a0"], delta=0.001)

    def test_transfer_learning_repvgg_a0_imagenet(self):
        trainer = SgModel('imagenet_pretrained_repvgg_a0_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("repvgg_a0", arch_params=self.imagenet_pretrained_arch_params["repvgg_a0"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_regseg48_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_regseg48', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset, data_loader_num_workers=8)
        trainer.build_model("regseg48", arch_params=self.cityscapes_pretrained_arch_params["regseg48"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.cityscapes_dataset.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["regseg48"], delta=0.001)

    def test_transfer_learning_regseg48_cityscapes(self):
        trainer = SgModel('regseg48_cityscapes_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("regseg48", arch_params=self.cityscapes_pretrained_arch_params["regseg48"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        trainer.train(training_params=self.regseg_transfer_segmentation_train_params)

    def test_pretrained_ddrnet23_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.cityscapes_dataset.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["ddrnet_23"], delta=0.001)

    def test_pretrained_ddrnet23_slim_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23_slim', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23_slim", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.cityscapes_dataset.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["ddrnet_23_slim"], delta=0.001)

    def test_transfer_learning_ddrnet23_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        trainer.train(training_params=self.ddrnet_transfer_segmentation_train_params)

    def test_transfer_learning_ddrnet23_slim_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_ddrnet23_slim_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("ddrnet_23_slim", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        trainer.train(training_params=self.ddrnet_transfer_segmentation_train_params)

    def test_pretrained_coco_segmentation_subclass_pretrained_shelfnet34_lw(self):
        trainer = SgModel('coco_segmentation_subclass_pretrained_shelfnet34_lw', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.coco_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("shelfnet34_lw",
                            arch_params=self.coco_segmentation_subclass_pretrained_arch_params["shelfnet34_lw"], checkpoint_params=self.coco_segmentation_subclass_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.coco_segmentation_dataset.val_loader, test_metrics_list=[IoU(21)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.coco_segmentation_subclass_pretrained_mious["shelfnet34_lw"], delta=0.001)

    def test_pretrained_efficientnet_b0_imagenet(self):
        trainer = SgModel('imagenet_pretrained_efficientnet_b0', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("efficientnet_b0", arch_params=self.imagenet_pretrained_arch_params["efficientnet_b0"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["efficientnet_b0"], delta=0.001)

    def test_transfer_learning_efficientnet_b0_imagenet(self):
        trainer = SgModel('imagenet_pretrained_efficientnet_b0_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("efficientnet_b0", arch_params=self.imagenet_pretrained_arch_params["efficientnet_b0"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_yolov5s_coco(self):
        trainer = SgModel('coco_pretrained_yolov5s', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.coco_dataset, data_loader_num_workers=8)
        trainer.build_model("yolo_v5s", arch_params=self.coco_pretrained_arch_params["yolo_v5"], checkpoint_params=self.coco_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.coco_dataset.val_loader,
                           test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloV5PostPredictionCallback(),
                                                               num_cls=len(
                                                                   self.coco_dataset.coco_classes))],
                           metrics_progress_verbose=True)[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolo_v5s"], delta=0.001)

    def test_pretrained_yolov5m_coco(self):
        trainer = SgModel('coco_pretrained_yolov5m', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.coco_dataset, data_loader_num_workers=8)
        trainer.build_model("yolo_v5m", arch_params=self.coco_pretrained_arch_params["yolo_v5"], checkpoint_params=self.coco_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.coco_dataset.val_loader,
                           test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloV5PostPredictionCallback(),
                                                               num_cls=len(
                                                                   self.coco_dataset.coco_classes))],
                           metrics_progress_verbose=True)[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolo_v5m"], delta=0.001)

    def test_pretrained_yolov5l_coco(self):
        trainer = SgModel('coco_pretrained_yolov5l', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.coco_dataset, data_loader_num_workers=8)
        trainer.build_model("yolo_v5l", arch_params=self.coco_pretrained_arch_params["yolo_v5"], checkpoint_params=self.coco_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.coco_dataset.val_loader,
                           test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloV5PostPredictionCallback(),
                                                               num_cls=len(
                                                                   self.coco_dataset.coco_classes))],
                           metrics_progress_verbose=True)[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolo_v5l"], delta=0.001)

    def test_pretrained_yolov5n_coco(self):
        trainer = SgModel('coco_pretrained_yolov5n', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.coco_dataset, data_loader_num_workers=8)
        trainer.build_model("yolo_v5n", arch_params=self.coco_pretrained_arch_params["yolo_v5"], checkpoint_params=self.coco_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.coco_dataset.val_loader,
                           test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloV5PostPredictionCallback(),
                                                               num_cls=len(
                                                                   self.coco_dataset.coco_classes))],
                           metrics_progress_verbose=True)[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolo_v5n"], delta=0.001)

    def test_transfer_learning_mobilenet_v3_large_imagenet(self):
        trainer = SgModel('imagenet_pretrained_mobilenet_v3_large_transfer_learning',
                          model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("mobilenet_v3_large", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_mobilenet_v3_large_imagenet(self):
        trainer = SgModel('imagenet_mobilenet_v3_large', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("mobilenet_v3_large", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["mobilenet_v3_large"], delta=0.001)

    def test_transfer_learning_mobilenet_v3_small_imagenet(self):
        trainer = SgModel('imagenet_pretrained_mobilenet_v3_small_transfer_learning',
                          model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("mobilenet_v3_small", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_mobilenet_v3_small_imagenet(self):
        trainer = SgModel('imagenet_mobilenet_v3_small', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("mobilenet_v3_small", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["mobilenet_v3_small"], delta=0.001)

    def test_transfer_learning_mobilenet_v2_imagenet(self):
        trainer = SgModel('imagenet_pretrained_mobilenet_v2_transfer_learning',
                          model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("mobilenet_v2", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_mobilenet_v2_imagenet(self):
        trainer = SgModel('imagenet_mobilenet_v2', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset, data_loader_num_workers=8)
        trainer.build_model("mobilenet_v2", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["mobilenet_v2"], delta=0.001)

    def test_pretrained_stdc1_seg50_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc1_seg50', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset_rescaled50, data_loader_num_workers=8)
        trainer.build_model("stdc1_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.cityscapes_dataset_rescaled50.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc1_seg50"], delta=0.001)

    def test_transfer_learning_stdc1_seg50_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc1_seg50_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("stdc1_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        trainer.train(training_params=self.stdc_transfer_segmentation_train_params)

    def test_pretrained_stdc1_seg75_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc1_seg75', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset_rescaled75, data_loader_num_workers=8)
        trainer.build_model("stdc1_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.cityscapes_dataset_rescaled75.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc1_seg75"], delta=0.001)

    def test_transfer_learning_stdc1_seg75_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc1_seg75_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("stdc1_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        trainer.train(training_params=self.stdc_transfer_segmentation_train_params)

    def test_pretrained_stdc2_seg50_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc2_seg50', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset_rescaled50, data_loader_num_workers=8)
        trainer.build_model("stdc2_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.cityscapes_dataset_rescaled50.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc2_seg50"], delta=0.001)

    def test_transfer_learning_stdc2_seg50_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc2_seg50_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("stdc2_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        trainer.train(training_params=self.stdc_transfer_segmentation_train_params)

    def test_pretrained_stdc2_seg75_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc2_seg75', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.cityscapes_dataset_rescaled75, data_loader_num_workers=8)
        trainer.build_model("stdc2_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.cityscapes_dataset_rescaled75.val_loader,
                           test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc2_seg75"], delta=0.001)

    def test_transfer_learning_stdc2_seg75_cityscapes(self):
        trainer = SgModel('cityscapes_pretrained_stdc2_seg75_transfer_learning', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_segmentation_dataset, data_loader_num_workers=8)
        trainer.build_model("stdc2_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], checkpoint_params=self.cityscapes_pretrained_ckpt_params)
        trainer.train(training_params=self.stdc_transfer_segmentation_train_params)

    def test_transfer_learning_vit_base_imagenet21k(self):
        trainer = SgModel('imagenet21k_pretrained_vit_base',
                          model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("vit_base", arch_params=self.imagenet_pretrained_arch_params["vit_base"],
                            checkpoint_params=self.imagenet21k_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_transfer_learning_vit_large_imagenet21k(self):
        trainer = SgModel('imagenet21k_pretrained_vit_large',
                          model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.transfer_classification_dataset, data_loader_num_workers=8)
        trainer.build_model("vit_large", arch_params=self.imagenet_pretrained_arch_params["vit_base"],
                            checkpoint_params=self.imagenet21k_pretrained_ckpt_params)
        trainer.train(training_params=self.transfer_classification_train_params)

    def test_pretrained_vit_base_imagenet(self):
        trainer = SgModel('imagenet_pretrained_vit_base', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset_05_mean_std, data_loader_num_workers=8)
        trainer.build_model("vit_base", arch_params=self.imagenet_pretrained_arch_params["vit_base"],
                            checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset_05_mean_std.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["vit_base"], delta=0.001)

    def test_pretrained_vit_large_imagenet(self):
        trainer = SgModel('imagenet_pretrained_vit_large', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset_05_mean_std, data_loader_num_workers=8)
        trainer.build_model("vit_large", arch_params=self.imagenet_pretrained_arch_params["vit_base"],
                            checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset_05_mean_std.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["vit_large"], delta=0.001)

    def test_pretrained_beit_base_imagenet(self):
        trainer = SgModel('imagenet_pretrained_beit_base', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)
        trainer.connect_dataset_interface(self.imagenet_dataset_05_mean_std, data_loader_num_workers=8)
        trainer.build_model("beit_base_patch16_224", arch_params=self.imagenet_pretrained_arch_params["vit_base"],
                            checkpoint_params=self.imagenet_pretrained_ckpt_params)
        res = trainer.test(test_loader=self.imagenet_dataset_05_mean_std.val_loader, test_metrics_list=[Accuracy()],
                           metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["beit_base_patch16_224"], delta=0.001)

    def tearDown(self) -> None:
        if os.path.exists('~/.cache/torch/hub/'):
            shutil.rmtree('~/.cache/torch/hub/')


if __name__ == '__main__':
    unittest.main()
