import unittest
from copy import deepcopy
from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader, detection_test_dataloader, segmentation_test_dataloader
from super_gradients.training.losses import PPYoloELoss, STDCLoss
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss
from super_gradients.training.metrics import Accuracy, DetectionMetrics, DetectionMetrics_050, IoU
from super_gradients.training.models import YoloXPostPredictionCallback
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training import models
from super_gradients.common.object_names import Models


class TestFineTune(unittest.TestCase):
    def test_train_with_finetune_customizable_detector(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_customizable_detector")
        net = models.get(Models.YOLO_NAS_S, num_classes=5, pretrained_weights="coco")
        net_before_train = deepcopy(net)

        train_params = {
            "initial_lr": 5e-4,
            "finetune": True,
            "lr_mode": "cosine",
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 0.0001},
            "max_epochs": 3,
            "mixed_precision": True,
            "average_best_models": False,
            "loss": PPYoloELoss(use_static_assigner=False, num_classes=5, reg_max=16),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=5,
                    normalize_targets=True,
                    include_classwise_ap=False,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
                )
            ],
            "metric_to_watch": "mAP@0.50",
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=detection_test_dataloader(),
            valid_loader=detection_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(net_before_train.backbone, net.backbone, skip_bn_stats=True))
        self.assertTrue(check_models_have_same_weights(net_before_train.neck, net.neck, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train.heads, net.heads))

    def test_train_with_finetune_ppyoloe(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_ppyoloe")
        net = models.get(Models.PP_YOLOE_S, num_classes=5, pretrained_weights="coco")
        net_before_train = deepcopy(net)

        train_params = {
            "initial_lr": 5e-4,
            "finetune": True,
            "lr_mode": "cosine",
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 0.0001},
            "max_epochs": 3,
            "mixed_precision": True,
            "average_best_models": False,
            "loss": PPYoloELoss(use_static_assigner=False, num_classes=5, reg_max=16),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=5,
                    normalize_targets=True,
                    include_classwise_ap=False,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
                )
            ],
            "metric_to_watch": "mAP@0.50",
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=detection_test_dataloader(),
            valid_loader=detection_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(net_before_train.backbone, net.backbone, skip_bn_stats=True))
        self.assertTrue(check_models_have_same_weights(net_before_train.neck, net.neck, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train.head, net.head))

    def test_train_with_finetune_yolox(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_yolox")
        net = models.get(Models.YOLOX_S, num_classes=5, pretrained_weights="coco")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "average_best_models": False,
            "initial_lr": 0.02,
            "loss": "YoloXDetectionLoss",
            "criterion_params": {"strides": [8, 16, 32], "num_classes": 5},  # output strides of all yolo outputs
            "train_metrics_list": [],
            "valid_metrics_list": [DetectionMetrics(post_prediction_callback=YoloXPostPredictionCallback(), normalize_targets=True, num_cls=5)],
            "metric_to_watch": "mAP@0.50:0.95",
            "greater_metric_to_watch_is_better": True,
            "finetune": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=detection_test_dataloader(),
            valid_loader=detection_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(net_before_train._backbone, net._backbone, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train._head, net._head))

    def test_train_with_finetune_ddrnet(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_ddrnet")
        net = models.get(Models.DDRNET_23, num_classes=5, pretrained_weights="cityscapes", arch_params={"use_aux_heads": True})
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "finetune": True,
            "loss": DDRNetLoss(),
            "lr_mode": "PolyLRScheduler",
            "ema": True,  # unlike the paper (not specified in paper)
            "average_best_models": False,
            "optimizer": "SGD",
            "mixed_precision": False,
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=segmentation_test_dataloader(),
            valid_loader=segmentation_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(net_before_train.final_layer, net.final_layer, skip_bn_stats=True))
        self.assertTrue(check_models_have_same_weights(net_before_train.seghead_extra, net.seghead_extra, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train, net))

    def test_train_with_finetune_ppliteseg(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_ppliteseg")
        net = models.get(Models.PP_LITE_T_SEG50, num_classes=5, pretrained_weights="cityscapes", arch_params={"use_aux_heads": True})
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "finetune": True,
            "loss": {
                "DiceCEEdgeLoss": {
                    "num_classes": 5,
                    "num_aux_heads": 3,
                    "num_detail_heads": 0,
                    "weights": [1.0, 1.0, 1.0, 1.0],
                    "dice_ce_weights": [1.0, 1.0],
                    "ce_edge_weights": [0.5, 0.5],
                    "edge_kernel": 5,
                }
            },
            "average_best_models": False,
            "optimizer": "SGD",
            "mixed_precision": False,
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=segmentation_test_dataloader(),
            valid_loader=segmentation_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(net_before_train.seg_head, net.seg_head, skip_bn_stats=True))
        self.assertTrue(check_models_have_same_weights(net_before_train.aux_heads, net.aux_heads, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train, net))

    def test_train_with_finetune_regseg(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_regseg")
        net = models.get(Models.REGSEG48, num_classes=5, pretrained_weights="cityscapes")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": "CrossEntropyLoss",
            "lr_mode": "PolyLRScheduler",
            "ema": True,
            "optimizer": "SGD",
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
            "finetune": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=segmentation_test_dataloader(),
            valid_loader=segmentation_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(net_before_train.stem, net.stem, skip_bn_stats=True))
        self.assertTrue(check_models_have_same_weights(net_before_train.backbone, net.backbone, skip_bn_stats=True))
        self.assertTrue(check_models_have_same_weights(net_before_train.decoder, net.decoder, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train.head, net.head))

    def test_train_with_finetune_segformer(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_segformer")
        net = models.get(Models.SEGFORMER_B0, num_classes=5, pretrained_weights="cityscapes")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": "CrossEntropyLoss",
            "lr_mode": "PolyLRScheduler",
            "ema": True,
            "optimizer": "SGD",
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=segmentation_test_dataloader(),
            valid_loader=segmentation_test_dataloader(),
        )
        self.assertTrue(check_models_have_same_weights(net_before_train._backbone, net._backbone, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train.decode_head, net.decode_head))

    def test_train_with_finetune_stdc(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_stdc")
        net = models.get(Models.STDC1_SEG50, num_classes=5, pretrained_weights="cityscapes")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": STDCLoss(num_classes=5),
            "lr_mode": "PolyLRScheduler",
            "ema": True,  # unlike the paper (not specified in paper)
            "optimizer": "SGD",
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=segmentation_test_dataloader(),
            valid_loader=segmentation_test_dataloader(),
        )
        self.assertTrue(check_models_have_same_weights(net_before_train.cp, net.cp, skip_bn_stats=True))
        self.assertTrue(check_models_have_same_weights(net_before_train.ffm, net.ffm, skip_bn_stats=True))

        self.assertFalse(check_models_have_same_weights(net_before_train.detail_head8, net.detail_head8))
        self.assertFalse(check_models_have_same_weights(net_before_train.aux_head_s16, net.aux_head_s16))
        self.assertFalse(check_models_have_same_weights(net_before_train.aux_head_s32, net.aux_head_s32))
        self.assertFalse(check_models_have_same_weights(net_before_train.segmentation_head, net.segmentation_head))

    def test_train_with_finetune_beit(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_beit")
        net = models.get(Models.BEIT_BASE_PATCH16_224, num_classes=5, pretrained_weights="imagenet")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "initial_lr": 0.6,
            "loss": "CrossEntropyLoss",
            "lr_mode": "StepLRScheduler",
            "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(image_size=224),
            valid_loader=classification_test_dataloader(image_size=224),
        )
        self.assertFalse(check_models_have_same_weights(net_before_train, net))
        self.assertTrue(check_models_have_same_weights(net_before_train.head, net.head, skip_bn_stats=True))

    def test_train_with_finetune_efficientnet(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_efficientnet")
        net = models.get(Models.EFFICIENTNET_B0, num_classes=5, pretrained_weights="imagenet")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "initial_lr": 0.6,
            "loss": "CrossEntropyLoss",
            "lr_mode": "StepLRScheduler",
            "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(image_size=224),
            valid_loader=classification_test_dataloader(image_size=224),
        )
        self.assertFalse(check_models_have_same_weights(net_before_train, net))
        self.assertTrue(check_models_have_same_weights(net_before_train._fc, net._fc, skip_bn_stats=True))

    def test_train_with_finetune_mobilenet(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_mobilenet")
        net = models.get(Models.MOBILENET_V3_SMALL, num_classes=5, pretrained_weights="imagenet")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "initial_lr": 0.6,
            "loss": "CrossEntropyLoss",
            "lr_mode": "StepLRScheduler",
            "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(image_size=224),
            valid_loader=classification_test_dataloader(image_size=224),
        )
        self.assertFalse(check_models_have_same_weights(net_before_train, net))
        self.assertTrue(check_models_have_same_weights(net_before_train.classifier, net.classifier, skip_bn_stats=True))

    def test_train_with_finetune_regenet(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_regenet")
        net = models.get(Models.REGNETY200, num_classes=5, pretrained_weights="imagenet")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "initial_lr": 0.6,
            "loss": "CrossEntropyLoss",
            "lr_mode": "StepLRScheduler",
            "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(image_size=224),
            valid_loader=classification_test_dataloader(image_size=224),
        )
        self.assertFalse(check_models_have_same_weights(net_before_train, net))
        self.assertTrue(check_models_have_same_weights(net_before_train.net.head, net.net.head, skip_bn_stats=True))

    def test_train_with_finetune_repvgg(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_repvgg")
        net = models.get(Models.REPVGG_A0, num_classes=5, pretrained_weights="imagenet", arch_params={"build_residual_branches": True})
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "initial_lr": 0.6,
            "loss": "CrossEntropyLoss",
            "lr_mode": "StepLRScheduler",
            "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(image_size=224),
            valid_loader=classification_test_dataloader(image_size=224),
        )
        self.assertFalse(check_models_have_same_weights(net_before_train, net))
        self.assertTrue(check_models_have_same_weights(net_before_train.linear, net.linear, skip_bn_stats=True))

    def test_train_with_finetune_resnet(self):
        # Define Model
        trainer = Trainer("test_train_with_finetune_resnet")
        net = models.get(Models.RESNET18, num_classes=5, pretrained_weights="imagenet")
        net_before_train = deepcopy(net)

        train_params = {
            "max_epochs": 3,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "initial_lr": 0.6,
            "loss": "CrossEntropyLoss",
            "lr_mode": "StepLRScheduler",
            "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(image_size=224),
            valid_loader=classification_test_dataloader(image_size=224),
        )
        self.assertFalse(check_models_have_same_weights(net_before_train, net))
        self.assertTrue(check_models_have_same_weights(net_before_train.linear, net.linear, skip_bn_stats=True))
