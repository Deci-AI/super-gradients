#!/usr/bin/env python3
import unittest

from super_gradients.common.object_names import Models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders import imagenet_val, imagenet_vit_base_val
from super_gradients.training.dataloaders.dataloaders import (
    classification_test_dataloader,
    coco2017_val_yolox,
    coco2017_val_ssd_lite_mobilenet_v2,
    detection_test_dataloader,
    coco_segmentation_val,
    cityscapes_val,
    cityscapes_stdc_seg50_val,
    cityscapes_stdc_seg75_val,
    segmentation_test_dataloader,
    coco2017_val_ppyoloe,
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN, CrowdDetectionPPYoloECollateFN

from super_gradients.training.metrics import Accuracy, IoU
import os
import shutil
from super_gradients.training.utils.ssd_utils import SSDPostPredictCallback
from super_gradients.training.models.detection_models.ssd import DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.losses.stdc_loss import STDCLoss
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training import models
import super_gradients


class PretrainedModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.imagenet_pretrained_arch_params = {
            "resnet": {},
            "regnet": {},
            Models.REPVGG_A0: {"build_residual_branches": True},
            Models.EFFICIENTNET_B0: {},
            "mobilenet": {},
            Models.VIT_BASE: {"image_size": (224, 224), "patch_size": (16, 16)},
        }

        self.imagenet_pretrained_trainsfer_learning_arch_params = {
            "resnet": {},
            "regnet": {},
            Models.REPVGG_A0: {"build_residual_branches": True},
            Models.EFFICIENTNET_B0: {},
            "mobilenet": {},
            Models.VIT_BASE: {"image_size": (224, 224), "patch_size": (16, 16)},
        }

        self.imagenet_pretrained_ckpt_params = {"pretrained_weights": "imagenet"}

        self.imagenet21k_pretrained_ckpt_params = {"pretrained_weights": "imagenet21k"}

        self.imagenet_pretrained_accuracies = {
            Models.RESNET50: 0.8191,
            Models.RESNET34: 0.7413,
            Models.RESNET18: 0.706,
            Models.REPVGG_A0: 0.7205,
            Models.REGNETY800: 0.7707,
            Models.REGNETY600: 0.7618,
            Models.REGNETY400: 0.7474,
            Models.REGNETY200: 0.7088,
            Models.EFFICIENTNET_B0: 0.7762,
            Models.MOBILENET_V3_LARGE: 0.7452,
            Models.MOBILENET_V3_SMALL: 0.6745,
            Models.MOBILENET_V2: 0.7308,
            Models.VIT_BASE: 0.8415,
            Models.VIT_LARGE: 0.8564,
            Models.BEIT_BASE_PATCH16_224: 0.85,
        }
        self.imagenet_dataset = imagenet_val(dataloader_params={"batch_size": 128})

        self.imagenet_dataset_05_mean_std = imagenet_vit_base_val(dataloader_params={"batch_size": 128})

        self.transfer_classification_dataloader = classification_test_dataloader(image_size=224)

        self.transfer_classification_train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "initial_lr": 0.6,
            "loss": "cross_entropy",
            "lr_mode": "step",
            "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        self.coco_pretrained_ckpt_params = {"pretrained_weights": "coco"}
        self.coco_pretrained_arch_params = {Models.SSD_LITE_MOBILENET_V2: {"num_classes": 80}, Models.SSD_MOBILENET_V1: {"num_classes": 80}}
        self.coco_pretrained_ckpt_params = {"pretrained_weights": "coco"}

        self.coco_dataset = {
            "yolox": coco2017_val_yolox(dataloader_params={"collate_fn": CrowdDetectionCollateFN()}, dataset_params={"with_crowd": True}),
            "ppyoloe": coco2017_val_ppyoloe(
                dataloader_params={"collate_fn": CrowdDetectionPPYoloECollateFN(), "batch_size": 1},
                dataset_params={"with_crowd": True, "ignore_empty_annotations": False},
            ),
            "ssd_mobilenet": coco2017_val_ssd_lite_mobilenet_v2(
                dataloader_params={"collate_fn": CrowdDetectionCollateFN()}, dataset_params={"with_crowd": True}
            ),
        }

        self.coco_pretrained_maps = {
            Models.SSD_LITE_MOBILENET_V2: 0.2041,
            Models.SSD_MOBILENET_V1: 0.243,
            Models.YOLOX_S: 0.4047,
            Models.YOLOX_M: 0.4640,
            Models.YOLOX_L: 0.4925,
            Models.YOLOX_N: 0.2677,
            Models.YOLOX_T: 0.3718,
            Models.PP_YOLOE_S: 0.4252,
            Models.PP_YOLOE_M: 0.4711,
            Models.PP_YOLOE_L: 0.4948,
            Models.PP_YOLOE_X: 0.5115,
        }

        self.transfer_detection_dataset = detection_test_dataloader()

        ssd_dboxes = DEFAULT_SSD_LITE_MOBILENET_V2_ARCH_PARAMS["heads"]["SSDHead"]["anchors"]
        self.transfer_detection_train_params_ssd = {
            "max_epochs": 3,
            "lr_mode": "cosine",
            "initial_lr": 0.01,
            "cosine_final_lr_ratio": 0.01,
            "lr_warmup_epochs": 3,
            "batch_accumulate": 1,
            "loss": "ssd_loss",
            "criterion_params": {"dboxes": ssd_dboxes},
            "optimizer": "SGD",
            "warmup_momentum": 0.8,
            "optimizer_params": {"momentum": 0.937, "weight_decay": 0.0005, "nesterov": True},
            "train_metrics_list": [],
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=SSDPostPredictCallback(), num_cls=5)],
            "metric_to_watch": "mAP@0.50:0.95",
            "greater_metric_to_watch_is_better": True,
        }
        self.transfer_detection_train_params_yolox = {
            "max_epochs": 3,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.05,
            "warmup_bias_lr": 0.0,
            "warmup_momentum": 0.9,
            "initial_lr": 0.02,
            "loss": "yolox_loss",
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 5},  # output strides of all yolo outputs
            "train_metrics_list": [],
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), normalize_targets=True, num_cls=5)],
            "metric_to_watch": "mAP@0.50:0.95",
            "greater_metric_to_watch_is_better": True,
        }

        self.coco_segmentation_subclass_pretrained_arch_params = {"shelfnet34_lw": {"num_classes": 21, "image_size": 512}}
        self.coco_segmentation_subclass_pretrained_ckpt_params = {"pretrained_weights": "coco_segmentation_subclass"}
        self.coco_segmentation_subclass_pretrained_mious = {"shelfnet34_lw": 0.651}
        self.coco_segmentation_dataset = coco_segmentation_val()

        self.cityscapes_pretrained_models = [Models.DDRNET_23, Models.DDRNET_23_SLIM, Models.STDC1_SEG50, Models.REGSEG48]
        self.cityscapes_pretrained_arch_params = {
            Models.DDRNET_23: {"use_aux_heads": True},
            Models.REGSEG48: {},
            "stdc": {"use_aux_heads": True},
            "pplite_seg": {"use_aux_heads": True},
        }

        self.cityscapes_pretrained_ckpt_params = {"pretrained_weights": "cityscapes"}
        self.cityscapes_pretrained_mious = {
            Models.DDRNET_39: 0.8517,
            Models.DDRNET_23: 0.8148,
            Models.DDRNET_23_SLIM: 0.7941,
            Models.STDC1_SEG50: 0.7511,
            Models.STDC1_SEG75: 0.7687,
            Models.STDC2_SEG50: 0.7644,
            Models.STDC2_SEG75: 0.7893,
            Models.REGSEG48: 0.7815,
            Models.PP_LITE_T_SEG50: 0.7492,
            Models.PP_LITE_T_SEG75: 0.7756,
            Models.PP_LITE_B_SEG50: 0.7648,
            Models.PP_LITE_B_SEG75: 0.7852,
        }

        self.cityscapes_dataset = cityscapes_val()

        self.cityscapes_dataset_rescaled50 = cityscapes_stdc_seg50_val()
        self.cityscapes_dataset_rescaled75 = cityscapes_stdc_seg75_val()

        self.transfer_segmentation_dataset = segmentation_test_dataloader(image_size=1024)
        self.ddrnet_transfer_segmentation_train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": DDRNetLoss(),
            "lr_mode": "poly",
            "ema": True,  # unlike the paper (not specified in paper)
            "average_best_models": True,
            "optimizer": "SGD",
            "mixed_precision": False,
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }

        self.stdc_transfer_segmentation_train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": STDCLoss(num_classes=5),
            "lr_mode": "poly",
            "ema": True,  # unlike the paper (not specified in paper)
            "optimizer": "SGD",
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }

        self.regseg_transfer_segmentation_train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": "cross_entropy",
            "lr_mode": "poly",
            "ema": True,  # unlike the paper (not specified in paper)
            "optimizer": "SGD",
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }

    def test_pretrained_resnet50_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet50")
        model = models.get(Models.RESNET50, arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.RESNET50], delta=0.001)

    def test_transfer_learning_resnet50_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet50_transfer_learning")
        model = models.get(Models.RESNET50, arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_resnet34_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet34")

        model = models.get(Models.RESNET34, arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.RESNET34], delta=0.001)

    def test_transfer_learning_resnet34_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet34_transfer_learning")
        model = models.get(Models.RESNET34, arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_resnet18_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet18")

        model = models.get(Models.RESNET18, arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.RESNET18], delta=0.001)

    def test_transfer_learning_resnet18_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet18_transfer_learning")
        model = models.get(Models.RESNET18, arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY800_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY800")

        model = models.get(Models.REGNETY800, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.REGNETY800], delta=0.001)

    def test_transfer_learning_regnetY800_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY800_transfer_learning")
        model = models.get(Models.REGNETY800, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY600_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY600")

        model = models.get(Models.REGNETY600, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.REGNETY600], delta=0.001)

    def test_transfer_learning_regnetY600_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY600_transfer_learning")
        model = models.get(Models.REGNETY600, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY400_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY400")

        model = models.get(Models.REGNETY400, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.REGNETY400], delta=0.001)

    def test_transfer_learning_regnetY400_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY400_transfer_learning")
        model = models.get(Models.REGNETY400, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY200_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY200")

        model = models.get(Models.REGNETY200, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.REGNETY200], delta=0.001)

    def test_transfer_learning_regnetY200_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY200_transfer_learning")
        model = models.get(Models.REGNETY200, arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_repvgg_a0")

        model = models.get(Models.REPVGG_A0, arch_params=self.imagenet_pretrained_arch_params[Models.REPVGG_A0], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.REPVGG_A0], delta=0.001)

    def test_transfer_learning_repvgg_a0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_repvgg_a0_transfer_learning")
        model = models.get(
            Models.REPVGG_A0, arch_params=self.imagenet_pretrained_arch_params[Models.REPVGG_A0], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regseg48_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_regseg48")
        model = models.get(Models.REGSEG48, arch_params=self.cityscapes_pretrained_arch_params[Models.REGSEG48], **self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(
            model=model, test_loader=self.cityscapes_dataset, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.REGSEG48], delta=0.001)

    def test_transfer_learning_regseg48_cityscapes(self):
        trainer = Trainer("regseg48_cityscapes_transfer_learning")
        model = models.get(Models.REGSEG48, arch_params=self.cityscapes_pretrained_arch_params[Models.REGSEG48], **self.cityscapes_pretrained_ckpt_params)
        trainer.train(
            model=model,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
            training_params=self.regseg_transfer_segmentation_train_params,
        )

    def test_pretrained_ddrnet23_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23")
        model = models.get(Models.DDRNET_23, arch_params=self.cityscapes_pretrained_arch_params[Models.DDRNET_23], **self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(
            model=model, test_loader=self.cityscapes_dataset, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.DDRNET_23], delta=0.001)

    def test_pretrained_ddrnet39_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet39")
        model = models.get(Models.DDRNET_39, arch_params=self.cityscapes_pretrained_arch_params[Models.DDRNET_23], **self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(
            model=model, test_loader=self.cityscapes_dataset, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.DDRNET_39], delta=0.001)

    def test_pretrained_ddrnet23_slim_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23_slim")
        model = models.get(
            Models.DDRNET_23_SLIM, arch_params=self.cityscapes_pretrained_arch_params[Models.DDRNET_23], **self.cityscapes_pretrained_ckpt_params
        )
        res = trainer.test(
            model=model, test_loader=self.cityscapes_dataset, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.DDRNET_23_SLIM], delta=0.001)

    def test_transfer_learning_ddrnet23_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23_transfer_learning")
        model = models.get(Models.DDRNET_23, arch_params=self.cityscapes_pretrained_arch_params[Models.DDRNET_23], **self.cityscapes_pretrained_ckpt_params)
        trainer.train(
            model=model,
            training_params=self.ddrnet_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_transfer_learning_ddrnet23_slim_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23_slim_transfer_learning")
        model = models.get(
            Models.DDRNET_23_SLIM, arch_params=self.cityscapes_pretrained_arch_params[Models.DDRNET_23], **self.cityscapes_pretrained_ckpt_params
        )
        trainer.train(
            model=model,
            training_params=self.ddrnet_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_pretrained_coco_segmentation_subclass_pretrained_shelfnet34_lw(self):
        trainer = Trainer("coco_segmentation_subclass_pretrained_shelfnet34_lw")
        model = models.get(
            "shelfnet34_lw",
            arch_params=self.coco_segmentation_subclass_pretrained_arch_params["shelfnet34_lw"],
            **self.coco_segmentation_subclass_pretrained_ckpt_params,
        )
        res = trainer.test(model=model, test_loader=self.coco_segmentation_dataset, test_metrics_list=[IoU(21)], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious["shelfnet34_lw"], delta=0.001)

    def test_pretrained_efficientnet_b0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_efficientnet_b0")

        model = models.get(
            Models.EFFICIENTNET_B0, arch_params=self.imagenet_pretrained_arch_params[Models.EFFICIENTNET_B0], **self.imagenet_pretrained_ckpt_params
        )
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.EFFICIENTNET_B0], delta=0.001)

    def test_transfer_learning_efficientnet_b0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_efficientnet_b0_transfer_learning")

        model = models.get(
            Models.EFFICIENTNET_B0,
            arch_params=self.imagenet_pretrained_arch_params[Models.EFFICIENTNET_B0],
            **self.imagenet_pretrained_ckpt_params,
            num_classes=5,
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_ssd_lite_mobilenet_v2_coco(self):
        trainer = Trainer("coco_ssd_lite_mobilenet_v2")
        model = models.get(
            Models.SSD_LITE_MOBILENET_V2, arch_params=self.coco_pretrained_arch_params[Models.SSD_LITE_MOBILENET_V2], **self.coco_pretrained_ckpt_params
        )
        ssd_post_prediction_callback = SSDPostPredictCallback()
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ssd_mobilenet"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=ssd_post_prediction_callback, num_cls=80)],
            metrics_progress_verbose=True,
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.SSD_LITE_MOBILENET_V2], delta=0.001)

    def test_transfer_learning_ssd_lite_mobilenet_v2_coco(self):
        trainer = Trainer("coco_ssd_lite_mobilenet_v2_transfer_learning")
        transfer_arch_params = self.coco_pretrained_arch_params[Models.SSD_LITE_MOBILENET_V2].copy()
        transfer_arch_params["num_classes"] = 5
        model = models.get(Models.SSD_LITE_MOBILENET_V2, arch_params=transfer_arch_params, **self.coco_pretrained_ckpt_params)
        trainer.train(
            model=model,
            training_params=self.transfer_detection_train_params_ssd,
            train_loader=self.transfer_detection_dataset,
            valid_loader=self.transfer_detection_dataset,
        )

    def test_pretrained_ssd_mobilenet_v1_coco(self):
        trainer = Trainer(Models.SSD_MOBILENET_V1)
        model = models.get(Models.SSD_MOBILENET_V1, arch_params=self.coco_pretrained_arch_params[Models.SSD_MOBILENET_V1], **self.coco_pretrained_ckpt_params)
        ssd_post_prediction_callback = SSDPostPredictCallback()
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ssd_mobilenet"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=ssd_post_prediction_callback, num_cls=80)],
            metrics_progress_verbose=True,
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.SSD_MOBILENET_V1], delta=0.001)

    def test_pretrained_yolox_s_coco(self):
        trainer = Trainer(Models.YOLOX_S)

        model = models.get(Models.YOLOX_S, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.YOLOX_S], delta=0.001)

    def test_pretrained_yolox_m_coco(self):
        trainer = Trainer(Models.YOLOX_M)
        model = models.get(Models.YOLOX_M, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.YOLOX_M], delta=0.001)

    def test_pretrained_yolox_l_coco(self):
        trainer = Trainer(Models.YOLOX_L)
        model = models.get(Models.YOLOX_L, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.YOLOX_L], delta=0.001)

    def test_pretrained_yolox_n_coco(self):
        trainer = Trainer(Models.YOLOX_N)

        model = models.get(Models.YOLOX_N, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.YOLOX_N], delta=0.001)

    def test_pretrained_yolox_t_coco(self):
        trainer = Trainer(Models.YOLOX_T)
        model = models.get(Models.YOLOX_T, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.YOLOX_T], delta=0.001)

    def test_pretrained_ppyoloe_s_coco(self):
        trainer = Trainer(Models.PP_YOLOE_S)

        model = models.get(Models.PP_YOLOE_S, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ppyoloe"],
            test_metrics_list=[
                DetectionMetrics(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=80,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
                )
            ],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.PP_YOLOE_S], delta=0.001)

    def test_pretrained_ppyoloe_m_coco(self):
        trainer = Trainer(Models.PP_YOLOE_M)

        model = models.get(Models.PP_YOLOE_M, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ppyoloe"],
            test_metrics_list=[
                DetectionMetrics(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=80,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
                )
            ],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.PP_YOLOE_M], delta=0.001)

    def test_pretrained_ppyoloe_l_coco(self):
        trainer = Trainer(Models.PP_YOLOE_L)

        model = models.get(Models.PP_YOLOE_L, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ppyoloe"],
            test_metrics_list=[
                DetectionMetrics(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=80,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
                )
            ],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.PP_YOLOE_L], delta=0.001)

    def test_pretrained_ppyoloe_x_coco(self):
        trainer = Trainer(Models.PP_YOLOE_X)

        model = models.get(Models.PP_YOLOE_X, **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ppyoloe"],
            test_metrics_list=[
                DetectionMetrics(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=80,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
                )
            ],
        )
        self.assertAlmostEqual(res["mAP@0.50:0.95"].cpu().item(), self.coco_pretrained_maps[Models.PP_YOLOE_X], delta=0.001)

    def test_transfer_learning_yolox_n_coco(self):
        trainer = Trainer("test_transfer_learning_yolox_n_coco")
        model = models.get(Models.YOLOX_N, **self.coco_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_detection_train_params_yolox,
            train_loader=self.transfer_detection_dataset,
            valid_loader=self.transfer_detection_dataset,
        )

    def test_transfer_learning_mobilenet_v3_large_imagenet(self):
        trainer = Trainer("imagenet_pretrained_mobilenet_v3_large_transfer_learning")

        model = models.get(
            Models.MOBILENET_V3_LARGE, arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_mobilenet_v3_large_imagenet(self):
        trainer = Trainer("imagenet_mobilenet_v3_large")

        model = models.get(Models.MOBILENET_V3_LARGE, arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.MOBILENET_V3_LARGE], delta=0.001)

    def test_transfer_learning_mobilenet_v3_small_imagenet(self):
        trainer = Trainer("imagenet_pretrained_mobilenet_v3_small_transfer_learning")

        model = models.get(
            Models.MOBILENET_V3_SMALL, arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_mobilenet_v3_small_imagenet(self):
        trainer = Trainer("imagenet_mobilenet_v3_small")

        model = models.get(Models.MOBILENET_V3_SMALL, arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.MOBILENET_V3_SMALL], delta=0.001)

    def test_transfer_learning_mobilenet_v2_imagenet(self):
        trainer = Trainer("imagenet_pretrained_mobilenet_v2_transfer_learning")

        model = models.get(
            Models.MOBILENET_V2, arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_mobilenet_v2_imagenet(self):
        trainer = Trainer("imagenet_mobilenet_v2")

        model = models.get(Models.MOBILENET_V2, arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.MOBILENET_V2], delta=0.001)

    def test_pretrained_stdc1_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg50")

        model = models.get(Models.STDC1_SEG50, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.cityscapes_dataset_rescaled50,
            test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
            metrics_progress_verbose=True,
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.STDC1_SEG50], delta=0.001)

    def test_transfer_learning_stdc1_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg50_transfer_learning")
        model = models.get(
            Models.STDC1_SEG50, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_pretrained_stdc1_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg75")
        model = models.get(Models.STDC1_SEG75, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.cityscapes_dataset_rescaled75,
            test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
            metrics_progress_verbose=True,
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.STDC1_SEG75], delta=0.001)

    def test_transfer_learning_stdc1_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg75_transfer_learning")
        model = models.get(
            Models.STDC1_SEG75, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_pretrained_stdc2_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg50")
        model = models.get(Models.STDC2_SEG50, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(
            model=model, test_loader=self.cityscapes_dataset_rescaled50, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.STDC2_SEG50], delta=0.001)

    def test_transfer_learning_stdc2_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg50_transfer_learning")
        model = models.get(
            Models.STDC2_SEG50, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_pretrained_stdc2_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg75")
        model = models.get(Models.STDC2_SEG75, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.cityscapes_dataset_rescaled75,
            test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
            metrics_progress_verbose=True,
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.STDC2_SEG75], delta=0.001)

    def test_transfer_learning_stdc2_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg75_transfer_learning")
        model = models.get(
            Models.STDC2_SEG75, arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_transfer_learning_vit_base_imagenet21k(self):
        trainer = Trainer("imagenet21k_pretrained_vit_base")

        model = models.get(
            Models.VIT_BASE, arch_params=self.imagenet_pretrained_arch_params[Models.VIT_BASE], **self.imagenet21k_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_transfer_learning_vit_large_imagenet21k(self):
        trainer = Trainer("imagenet21k_pretrained_vit_large")

        model = models.get(
            Models.VIT_LARGE, arch_params=self.imagenet_pretrained_arch_params[Models.VIT_BASE], **self.imagenet21k_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_vit_base_imagenet(self):
        trainer = Trainer("imagenet_pretrained_vit_base")
        model = models.get(Models.VIT_BASE, arch_params=self.imagenet_pretrained_arch_params[Models.VIT_BASE], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset_05_mean_std, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.VIT_BASE], delta=0.001)

    def test_pretrained_vit_large_imagenet(self):
        trainer = Trainer("imagenet_pretrained_vit_large")
        model = models.get(Models.VIT_LARGE, arch_params=self.imagenet_pretrained_arch_params[Models.VIT_BASE], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset_05_mean_std, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.VIT_LARGE], delta=0.001)

    def test_pretrained_beit_base_imagenet(self):
        trainer = Trainer("imagenet_pretrained_beit_base")
        model = models.get(
            Models.BEIT_BASE_PATCH16_224, arch_params=self.imagenet_pretrained_arch_params[Models.VIT_BASE], **self.imagenet_pretrained_ckpt_params
        )
        res = trainer.test(model=model, test_loader=self.imagenet_dataset_05_mean_std, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)
        self.assertAlmostEqual(res["Accuracy"].cpu().item(), self.imagenet_pretrained_accuracies[Models.BEIT_BASE_PATCH16_224], delta=0.001)

    def test_transfer_learning_beit_base_imagenet(self):
        trainer = Trainer("test_transfer_learning_beit_base_imagenet")

        model = models.get(
            Models.BEIT_BASE_PATCH16_224,
            arch_params=self.imagenet_pretrained_arch_params[Models.VIT_BASE],
            **self.imagenet_pretrained_ckpt_params,
            num_classes=5,
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_pplite_t_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_t_seg50")
        model = models.get(Models.PP_LITE_T_SEG50, arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = trainer.test(
            model=model, test_loader=self.cityscapes_dataset_rescaled50, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.PP_LITE_T_SEG50], delta=0.001)

    def test_pretrained_pplite_t_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_t_seg75")
        model = models.get(Models.PP_LITE_T_SEG75, arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = trainer.test(
            model=model, test_loader=self.cityscapes_dataset_rescaled50, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.PP_LITE_T_SEG75], delta=0.001)

    def test_pretrained_pplite_b_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_b_seg50")
        model = models.get(Models.PP_LITE_B_SEG50, arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = trainer.test(
            model=model,
            test_loader=self.cityscapes_dataset_rescaled50,
            test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
            metrics_progress_verbose=True,
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.PP_LITE_B_SEG50], delta=0.001)

    def test_pretrained_pplite_b_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_b_seg75")
        model = models.get(Models.PP_LITE_B_SEG75, arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = trainer.test(
            model=model,
            test_loader=self.cityscapes_dataset_rescaled50,
            test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
            metrics_progress_verbose=True,
        )
        self.assertAlmostEqual(res["IoU"].cpu().item(), self.cityscapes_pretrained_mious[Models.PP_LITE_B_SEG75], delta=0.001)

    def tearDown(self) -> None:
        if os.path.exists("~/.cache/torch/hub/"):
            shutil.rmtree("~/.cache/torch/hub/")


if __name__ == "__main__":
    unittest.main()
