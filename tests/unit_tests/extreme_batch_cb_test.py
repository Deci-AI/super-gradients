import unittest
from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import segmentation_test_dataloader
from super_gradients.training.losses.ddrnet_loss import DDRNetLoss
from super_gradients.training.metrics import IoU
from super_gradients.training.utils.callbacks.callbacks import ExtremeBatchSegVisualizationCallback


# Helper method to set up Trainer and model with common parameters
def setup_trainer_and_model(experiment_name: str):
    trainer = Trainer(experiment_name)
    model = models.get(Models.DDRNET_23, arch_params={"use_aux_heads": True}, pretrained_weights="cityscapes")
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
        cls.training_params = {
            "max_epochs": 3,
            "initial_lr": 1e-2,
            "loss": DDRNetLoss(),
            "lr_mode": "poly",
            "ema": True,
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

    def test_segmentation_extreme_batch_with_metric_sanity(self):
        trainer, model = setup_trainer_and_model("test_segmentation_extreme_batch_with_metric_sanity")
        self.training_params["phase_callbacks"] = [ExtremeBatchSegVisualizationCallback(IoU(5))]
        trainer.train(
            model=model, training_params=self.training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )

    def test_segmentation_extreme_batch_with_compound_metric_sanity(self):
        trainer, model = setup_trainer_and_model("test_segmentation_extreme_batch_with_compound_metric_sanity")
        self.training_params["phase_callbacks"] = [ExtremeBatchSegVisualizationCallback(DummyIOU(5), metric_component_name="diou_minus")]
        trainer.train(
            model=model, training_params=self.training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )

    def test_segmentation_extreme_batch_with_loss_sanity(self):
        trainer, model = setup_trainer_and_model("test_segmentation_extreme_batch_with_loss_sanity")
        self.training_params["phase_callbacks"] = [ExtremeBatchSegVisualizationCallback(loss_to_monitor="DDRNetLoss/aux_loss1")]
        trainer.train(
            model=model, training_params=self.training_params, train_loader=segmentation_test_dataloader(), valid_loader=segmentation_test_dataloader()
        )


if __name__ == "__main__":
    unittest.main()
