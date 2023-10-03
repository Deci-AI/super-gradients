import unittest
from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import segmentation_test_dataloader, detection_test_dataloader
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss
from super_gradients.training.metrics import IoU, DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.callbacks.callbacks import ExtremeBatchSegVisualizationCallback, ExtremeBatchDetectionVisualizationCallback


# Helper method to set up Trainer and model with common parameters
def setup_trainer_and_model_seg(experiment_name: str):
    trainer = Trainer(experiment_name)
    model = models.get(Models.DDRNET_23, arch_params={"use_aux_heads": True}, pretrained_weights="cityscapes")
    return trainer, model


def setup_trainer_and_model_detection(experiment_name: str):
    trainer = Trainer(experiment_name)
    model = models.get(Models.YOLO_NAS_S, num_classes=1)
    return trainer, model


class DummyIOU(IoU):
    """
    Metric for testing the segmentation callback works with compound metrics
    """

    def compute(self):
        diou = super(DummyIOU, self).compute()
        return {"diou": diou, "diou_minus": -1 * diou}


class ExtremeBatchSanityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seg_training_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": DDRNetLoss(),
            "lr_mode": "PolyLRScheduler",
            "ema": True,
            "optimizer": "SGD",
            "mixed_precision": False,
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "train_metrics_list": [IoU(5)],
            "valid_metrics_list": [IoU(5)],
            "metric_to_watch": "IoU",
            "greater_metric_to_watch_is_better": True,
        }

        cls.od_training_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": PPYoloELoss(num_classes=1, use_static_assigner=False, reg_max=16),
            "lr_mode": "PolyLRScheduler",
            "ema": True,
            "optimizer": "SGD",
            "mixed_precision": False,
            "optimizer_params": {"weight_decay": 5e-4, "momentum": 0.9},
            "load_opt_params": False,
            "valid_metrics_list": [
                DetectionMetrics_050(
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
                    num_cls=1,
                )
            ],
            "train_metrics_list": [],
            "metric_to_watch": "mAP@0.50",
            "greater_metric_to_watch_is_better": True,
        }

    def test_detection_extreme_batch_with_metric_sanity(self):
        trainer, model = setup_trainer_and_model_detection("test_detection_extreme_batch_with_metric_sanity")
        self.od_training_params["phase_callbacks"] = [
            ExtremeBatchDetectionVisualizationCallback(
                classes=["1"],
                metric=DetectionMetrics_050(
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
                    num_cls=1,
                ),
                metric_component_name="mAP@0.50",
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
            )
        ]
        trainer.train(model=model, training_params=self.od_training_params, train_loader=detection_test_dataloader(), valid_loader=detection_test_dataloader())

    def test_detection_extreme_batch_with_loss_sanity(self):
        trainer, model = setup_trainer_and_model_detection("test_detection_extreme_batch_with_loss_sanity")
        self.od_training_params["phase_callbacks"] = [
            ExtremeBatchDetectionVisualizationCallback(
                classes=["1"],
                loss_to_monitor="PPYoloELoss/loss_cls",
                post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.03, nms_top_k=1000, max_predictions=300, nms_threshold=0.65),
            )
        ]
        trainer.train(model=model, training_params=self.od_training_params, train_loader=detection_test_dataloader(), valid_loader=detection_test_dataloader())

    def test_segmentation_extreme_batch_with_metric_sanity(self):
        trainer, model = setup_trainer_and_model_seg("test_segmentation_extreme_batch_with_metric_sanity")
        self.seg_training_params["phase_callbacks"] = [ExtremeBatchSegVisualizationCallback(IoU(5))]
        trainer.train(
            model=model, training_params=self.seg_training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )

    def test_segmentation_extreme_batch_with_compound_metric_sanity(self):
        trainer, model = setup_trainer_and_model_seg("test_segmentation_extreme_batch_with_compound_metric_sanity")
        self.seg_training_params["phase_callbacks"] = [ExtremeBatchSegVisualizationCallback(DummyIOU(5), metric_component_name="diou_minus")]
        trainer.train(
            model=model, training_params=self.seg_training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )

    def test_segmentation_extreme_batch_with_loss_sanity(self):
        trainer, model = setup_trainer_and_model_seg("test_segmentation_extreme_batch_with_loss_sanity")
        self.seg_training_params["phase_callbacks"] = [ExtremeBatchSegVisualizationCallback(loss_to_monitor="DDRNetLoss/aux_loss1")]
        trainer.train(
            model=model, training_params=self.seg_training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )

    def test_segmentation_extreme_batch_train_only(self):
        trainer, model = setup_trainer_and_model_seg("test_segmentation_extreme_batch_train_only")
        self.seg_training_params["phase_callbacks"] = [
            ExtremeBatchSegVisualizationCallback(loss_to_monitor="DDRNetLoss/aux_loss1", enable_on_train_loader=True, enable_on_valid_loader=False)
        ]
        trainer.train(
            model=model, training_params=self.seg_training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )

    def test_segmentation_extreme_batch_train_and_valid(self):
        trainer, model = setup_trainer_and_model_seg("test_segmentation_extreme_batch_train_and_valid")
        self.seg_training_params["phase_callbacks"] = [
            ExtremeBatchSegVisualizationCallback(loss_to_monitor="DDRNetLoss/aux_loss1", enable_on_train_loader=True, enable_on_valid_loader=True)
        ]
        trainer.train(
            model=model, training_params=self.seg_training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )


if __name__ == "__main__":
    unittest.main()
