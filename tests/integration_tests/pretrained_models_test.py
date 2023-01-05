#!/usr/bin/env python3
import unittest

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
)
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN

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
            "repvgg_a0": {"build_residual_branches": True},
            "efficientnet_b0": {},
            "mobilenet": {},
            "vit_base": {"image_size": (224, 224), "patch_size": (16, 16)},
        }

        self.imagenet_pretrained_trainsfer_learning_arch_params = {
            "resnet": {},
            "regnet": {},
            "repvgg_a0": {"build_residual_branches": True},
            "efficientnet_b0": {},
            "mobilenet": {},
            "vit_base": {"image_size": (224, 224), "patch_size": (16, 16)},
        }

        self.imagenet_pretrained_ckpt_params = {"pretrained_weights": "imagenet"}

        self.imagenet21k_pretrained_ckpt_params = {"pretrained_weights": "imagenet21k"}

        self.imagenet_pretrained_accuracies = {
            "resnet50": 0.8191,
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
            "beit_base_patch16_224": 0.85,
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
        self.coco_pretrained_arch_params = {"ssd_lite_mobilenet_v2": {"num_classes": 80}, "coco_ssd_mobilenet_v1": {"num_classes": 80}}
        self.coco_pretrained_ckpt_params = {"pretrained_weights": "coco"}

        self.coco_dataset = {
            "yolox": coco2017_val_yolox(dataloader_params={"collate_fn": CrowdDetectionCollateFN()}, dataset_params={"with_crowd": True}),
            "ssd_mobilenet": coco2017_val_ssd_lite_mobilenet_v2(
                dataloader_params={"collate_fn": CrowdDetectionCollateFN()}, dataset_params={"with_crowd": True}
            ),
        }

        self.coco_pretrained_maps = {
            "ssd_lite_mobilenet_v2": 0.2041,
            "coco_ssd_mobilenet_v1": 0.243,
            "yolox_s": 0.4047,
            "yolox_m": 0.4640,
            "yolox_l": 0.4925,
            "yolox_n": 0.2677,
            "yolox_t": 0.3718,
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

        self.cityscapes_pretrained_models = ["ddrnet_23", "ddrnet_23_slim", "stdc1_seg50", "regseg48"]
        self.cityscapes_pretrained_arch_params = {
            "ddrnet_23": {"aux_head": True},
            "regseg48": {},
            "stdc": {"use_aux_heads": True, "aux_head": True},
            "pplite_seg": {"use_aux_heads": True},
        }

        self.cityscapes_pretrained_ckpt_params = {"pretrained_weights": "cityscapes"}
        self.cityscapes_pretrained_mious = {
            "ddrnet_23": 0.8026,
            "ddrnet_23_slim": 0.7801,
            "stdc1_seg50": 0.7511,
            "stdc1_seg75": 0.7687,
            "stdc2_seg50": 0.7644,
            "stdc2_seg75": 0.7893,
            "regseg48": 0.7815,
            "pp_lite_t_seg50": 0.7492,
            "pp_lite_t_seg75": 0.7756,
            "pp_lite_b_seg50": 0.7648,
            "pp_lite_b_seg75": 0.7852,
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
        model = models.get("resnet50", arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet50"], delta=0.001)

    def test_transfer_learning_resnet50_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet50_transfer_learning")
        model = models.get("resnet50", arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_resnet34_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet34")

        model = models.get("resnet34", arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet34"], delta=0.001)

    def test_transfer_learning_resnet34_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet34_transfer_learning")
        model = models.get("resnet34", arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_resnet18_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet18")

        model = models.get("resnet18", arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["resnet18"], delta=0.001)

    def test_transfer_learning_resnet18_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet18_transfer_learning")
        model = models.get("resnet18", arch_params=self.imagenet_pretrained_arch_params["resnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY800_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY800")

        model = models.get("regnetY800", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY800"], delta=0.001)

    def test_transfer_learning_regnetY800_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY800_transfer_learning")
        model = models.get("regnetY800", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY600_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY600")

        model = models.get("regnetY600", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY600"], delta=0.001)

    def test_transfer_learning_regnetY600_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY600_transfer_learning")
        model = models.get("regnetY600", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY400_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY400")

        model = models.get("regnetY400", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY400"], delta=0.001)

    def test_transfer_learning_regnetY400_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY400_transfer_learning")
        model = models.get("regnetY400", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regnetY200_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY200")

        model = models.get("regnetY200", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["regnetY200"], delta=0.001)

    def test_transfer_learning_regnetY200_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY200_transfer_learning")
        model = models.get("regnetY200", arch_params=self.imagenet_pretrained_arch_params["regnet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_repvgg_a0")

        model = models.get("repvgg_a0", arch_params=self.imagenet_pretrained_arch_params["repvgg_a0"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["repvgg_a0"], delta=0.001)

    def test_transfer_learning_repvgg_a0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_repvgg_a0_transfer_learning")
        model = models.get("repvgg_a0", arch_params=self.imagenet_pretrained_arch_params["repvgg_a0"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_regseg48_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_regseg48")
        model = models.get("regseg48", arch_params=self.cityscapes_pretrained_arch_params["regseg48"], **self.cityscapes_pretrained_ckpt_params)
        res = (
            trainer.test(
                model=model, test_loader=self.cityscapes_dataset, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["regseg48"], delta=0.001)

    def test_transfer_learning_regseg48_cityscapes(self):
        trainer = Trainer("regseg48_cityscapes_transfer_learning")
        model = models.get("regseg48", arch_params=self.cityscapes_pretrained_arch_params["regseg48"], **self.cityscapes_pretrained_ckpt_params)
        trainer.train(
            model=model,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
            training_params=self.regseg_transfer_segmentation_train_params,
        )

    def test_pretrained_ddrnet23_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23")
        model = models.get("ddrnet_23", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], **self.cityscapes_pretrained_ckpt_params)
        res = (
            trainer.test(
                model=model, test_loader=self.cityscapes_dataset, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["ddrnet_23"], delta=0.001)

    def test_pretrained_ddrnet23_slim_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23_slim")
        model = models.get("ddrnet_23_slim", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], **self.cityscapes_pretrained_ckpt_params)
        res = (
            trainer.test(
                model=model, test_loader=self.cityscapes_dataset, test_metrics_list=[IoU(num_classes=20, ignore_index=19)], metrics_progress_verbose=True
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["ddrnet_23_slim"], delta=0.001)

    def test_transfer_learning_ddrnet23_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23_transfer_learning")
        model = models.get("ddrnet_23", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], **self.cityscapes_pretrained_ckpt_params)
        trainer.train(
            model=model,
            training_params=self.ddrnet_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_transfer_learning_ddrnet23_slim_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_ddrnet23_slim_transfer_learning")
        model = models.get("ddrnet_23_slim", arch_params=self.cityscapes_pretrained_arch_params["ddrnet_23"], **self.cityscapes_pretrained_ckpt_params)
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
        res = trainer.test(model=model, test_loader=self.coco_segmentation_dataset, test_metrics_list=[IoU(21)], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.coco_segmentation_subclass_pretrained_mious["shelfnet34_lw"], delta=0.001)

    def test_pretrained_efficientnet_b0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_efficientnet_b0")

        model = models.get("efficientnet_b0", arch_params=self.imagenet_pretrained_arch_params["efficientnet_b0"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["efficientnet_b0"], delta=0.001)

    def test_transfer_learning_efficientnet_b0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_efficientnet_b0_transfer_learning")

        model = models.get(
            "efficientnet_b0", arch_params=self.imagenet_pretrained_arch_params["efficientnet_b0"], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_ssd_lite_mobilenet_v2_coco(self):
        trainer = Trainer("coco_ssd_lite_mobilenet_v2")
        model = models.get("ssd_lite_mobilenet_v2", arch_params=self.coco_pretrained_arch_params["ssd_lite_mobilenet_v2"], **self.coco_pretrained_ckpt_params)
        ssd_post_prediction_callback = SSDPostPredictCallback()
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ssd_mobilenet"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=ssd_post_prediction_callback, num_cls=80)],
            metrics_progress_verbose=True,
        )[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["ssd_lite_mobilenet_v2"], delta=0.001)

    def test_transfer_learning_ssd_lite_mobilenet_v2_coco(self):
        trainer = Trainer("coco_ssd_lite_mobilenet_v2_transfer_learning")
        transfer_arch_params = self.coco_pretrained_arch_params["ssd_lite_mobilenet_v2"].copy()
        transfer_arch_params["num_classes"] = 5
        model = models.get("ssd_lite_mobilenet_v2", arch_params=transfer_arch_params, **self.coco_pretrained_ckpt_params)
        trainer.train(
            model=model,
            training_params=self.transfer_detection_train_params_ssd,
            train_loader=self.transfer_detection_dataset,
            valid_loader=self.transfer_detection_dataset,
        )

    def test_pretrained_ssd_mobilenet_v1_coco(self):
        trainer = Trainer("coco_ssd_mobilenet_v1")
        model = models.get("ssd_mobilenet_v1", arch_params=self.coco_pretrained_arch_params["coco_ssd_mobilenet_v1"], **self.coco_pretrained_ckpt_params)
        ssd_post_prediction_callback = SSDPostPredictCallback()
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["ssd_mobilenet"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=ssd_post_prediction_callback, num_cls=80)],
            metrics_progress_verbose=True,
        )[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["coco_ssd_mobilenet_v1"], delta=0.001)

    def test_pretrained_yolox_s_coco(self):
        trainer = Trainer("yolox_s")

        model = models.get("yolox_s", **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolox_s"], delta=0.001)

    def test_pretrained_yolox_m_coco(self):
        trainer = Trainer("yolox_m")
        model = models.get("yolox_m", **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolox_m"], delta=0.001)

    def test_pretrained_yolox_l_coco(self):
        trainer = Trainer("yolox_l")
        model = models.get("yolox_l", **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolox_l"], delta=0.001)

    def test_pretrained_yolox_n_coco(self):
        trainer = Trainer("yolox_n")

        model = models.get("yolox_n", **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolox_n"], delta=0.001)

    def test_pretrained_yolox_t_coco(self):
        trainer = Trainer("yolox_t")
        model = models.get("yolox_t", **self.coco_pretrained_ckpt_params)
        res = trainer.test(
            model=model,
            test_loader=self.coco_dataset["yolox"],
            test_metrics_list=[DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=80, normalize_targets=True)],
        )[2]
        self.assertAlmostEqual(res, self.coco_pretrained_maps["yolox_t"], delta=0.001)

    def test_transfer_learning_yolox_n_coco(self):
        trainer = Trainer("test_transfer_learning_yolox_n_coco")
        model = models.get("yolox_n", **self.coco_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_detection_train_params_yolox,
            train_loader=self.transfer_detection_dataset,
            valid_loader=self.transfer_detection_dataset,
        )

    def test_transfer_learning_mobilenet_v3_large_imagenet(self):
        trainer = Trainer("imagenet_pretrained_mobilenet_v3_large_transfer_learning")

        model = models.get(
            "mobilenet_v3_large", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_mobilenet_v3_large_imagenet(self):
        trainer = Trainer("imagenet_mobilenet_v3_large")

        model = models.get("mobilenet_v3_large", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["mobilenet_v3_large"], delta=0.001)

    def test_transfer_learning_mobilenet_v3_small_imagenet(self):
        trainer = Trainer("imagenet_pretrained_mobilenet_v3_small_transfer_learning")

        model = models.get(
            "mobilenet_v3_small", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_mobilenet_v3_small_imagenet(self):
        trainer = Trainer("imagenet_mobilenet_v3_small")

        model = models.get("mobilenet_v3_small", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["mobilenet_v3_small"], delta=0.001)

    def test_transfer_learning_mobilenet_v2_imagenet(self):
        trainer = Trainer("imagenet_pretrained_mobilenet_v2_transfer_learning")

        model = models.get("mobilenet_v2", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_mobilenet_v2_imagenet(self):
        trainer = Trainer("imagenet_mobilenet_v2")

        model = models.get("mobilenet_v2", arch_params=self.imagenet_pretrained_arch_params["mobilenet"], **self.imagenet_pretrained_ckpt_params)
        res = trainer.test(model=model, test_loader=self.imagenet_dataset, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0].cpu().item()
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["mobilenet_v2"], delta=0.001)

    def test_pretrained_stdc1_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg50")
        model = models.get("stdc1_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled50,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc1_seg50"], delta=0.001)

    def test_transfer_learning_stdc1_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg50_transfer_learning")
        model = models.get("stdc1_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_pretrained_stdc1_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg75")
        model = models.get("stdc1_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled75,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc1_seg75"], delta=0.001)

    def test_transfer_learning_stdc1_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc1_seg75_transfer_learning")
        model = models.get("stdc1_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_pretrained_stdc2_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg50")
        model = models.get("stdc2_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled50,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc2_seg50"], delta=0.001)

    def test_transfer_learning_stdc2_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg50_transfer_learning")
        model = models.get("stdc2_seg50", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_pretrained_stdc2_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg75")
        model = models.get("stdc2_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params)
        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled75,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["stdc2_seg75"], delta=0.001)

    def test_transfer_learning_stdc2_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_stdc2_seg75_transfer_learning")
        model = models.get("stdc2_seg75", arch_params=self.cityscapes_pretrained_arch_params["stdc"], **self.cityscapes_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.stdc_transfer_segmentation_train_params,
            train_loader=self.transfer_segmentation_dataset,
            valid_loader=self.transfer_segmentation_dataset,
        )

    def test_transfer_learning_vit_base_imagenet21k(self):
        trainer = Trainer("imagenet21k_pretrained_vit_base")

        model = models.get("vit_base", arch_params=self.imagenet_pretrained_arch_params["vit_base"], **self.imagenet21k_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_transfer_learning_vit_large_imagenet21k(self):
        trainer = Trainer("imagenet21k_pretrained_vit_large")

        model = models.get("vit_large", arch_params=self.imagenet_pretrained_arch_params["vit_base"], **self.imagenet21k_pretrained_ckpt_params, num_classes=5)
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_vit_base_imagenet(self):
        trainer = Trainer("imagenet_pretrained_vit_base")
        model = models.get("vit_base", arch_params=self.imagenet_pretrained_arch_params["vit_base"], **self.imagenet_pretrained_ckpt_params)
        res = (
            trainer.test(model=model, test_loader=self.imagenet_dataset_05_mean_std, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["vit_base"], delta=0.001)

    def test_pretrained_vit_large_imagenet(self):
        trainer = Trainer("imagenet_pretrained_vit_large")
        model = models.get("vit_large", arch_params=self.imagenet_pretrained_arch_params["vit_base"], **self.imagenet_pretrained_ckpt_params)
        res = (
            trainer.test(model=model, test_loader=self.imagenet_dataset_05_mean_std, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["vit_large"], delta=0.001)

    def test_pretrained_beit_base_imagenet(self):
        trainer = Trainer("imagenet_pretrained_beit_base")
        model = models.get("beit_base_patch16_224", arch_params=self.imagenet_pretrained_arch_params["vit_base"], **self.imagenet_pretrained_ckpt_params)
        res = (
            trainer.test(model=model, test_loader=self.imagenet_dataset_05_mean_std, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.imagenet_pretrained_accuracies["beit_base_patch16_224"], delta=0.001)

    def test_transfer_learning_beit_base_imagenet(self):
        trainer = Trainer("test_transfer_learning_beit_base_imagenet")

        model = models.get(
            "beit_base_patch16_224", arch_params=self.imagenet_pretrained_arch_params["vit_base"], **self.imagenet_pretrained_ckpt_params, num_classes=5
        )
        trainer.train(
            model=model,
            training_params=self.transfer_classification_train_params,
            train_loader=self.transfer_classification_dataloader,
            valid_loader=self.transfer_classification_dataloader,
        )

    def test_pretrained_pplite_t_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_t_seg50")
        model = models.get("pp_lite_t_seg50", arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled50,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["pp_lite_t_seg50"], delta=0.001)

    def test_pretrained_pplite_t_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_t_seg75")
        model = models.get("pp_lite_t_seg75", arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled50,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["pp_lite_t_seg75"], delta=0.001)

    def test_pretrained_pplite_b_seg50_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_b_seg50")
        model = models.get("pp_lite_b_seg50", arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled50,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["pp_lite_b_seg50"], delta=0.001)

    def test_pretrained_pplite_b_seg75_cityscapes(self):
        trainer = Trainer("cityscapes_pretrained_pplite_b_seg75")
        model = models.get("pp_lite_b_seg75", arch_params=self.cityscapes_pretrained_arch_params["pplite_seg"], **self.cityscapes_pretrained_ckpt_params)

        res = (
            trainer.test(
                model=model,
                test_loader=self.cityscapes_dataset_rescaled50,
                test_metrics_list=[IoU(num_classes=20, ignore_index=19)],
                metrics_progress_verbose=True,
            )[0]
            .cpu()
            .item()
        )
        self.assertAlmostEqual(res, self.cityscapes_pretrained_mious["pp_lite_b_seg75"], delta=0.001)

    def tearDown(self) -> None:
        if os.path.exists("~/.cache/torch/hub/"):
            shutil.rmtree("~/.cache/torch/hub/")


if __name__ == "__main__":
    unittest.main()
