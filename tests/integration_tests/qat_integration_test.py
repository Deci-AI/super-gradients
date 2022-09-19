import unittest

from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training import Trainer, MultiGPUMode, models
from super_gradients.training.metrics.classification_metrics import Accuracy
import os
from super_gradients.training.utils.quantization_utils import PostQATConversionCallback


class QATIntegrationTest(unittest.TestCase):
    def _get_trainer(self, experiment_name):
        trainer = Trainer(experiment_name,

                          multi_gpu=MultiGPUMode.OFF)
        model = models.get("resnet18", pretrained_weights="imagenet")
        return trainer, model

    def _get_train_params(self, qat_params):
        train_params = {"max_epochs": 2,
                        "lr_mode": "step",
                        "optimizer": "SGD",
                        "lr_updates": [],
                        "lr_decay_factor": 0.1,
                        "initial_lr": 0.001, "loss": "cross_entropy",
                        "train_metrics_list": [Accuracy()],
                        "valid_metrics_list": [Accuracy()],
                        "loss_logging_items_names": ["Loss"],
                        "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True,
                        "average_best_models": False,
                        "enable_qat": True,
                        "qat_params": qat_params,
                        "phase_callbacks": [PostQATConversionCallback(dummy_input_size=(1, 3, 224, 224))]
                        }
        return train_params

    def test_qat_from_start(self):
        model, net = self._get_trainer("test_qat_from_start")
        train_params = self._get_train_params(qat_params={
            "start_epoch": 0,
            "quant_modules_calib_method": "percentile",
            "calibrate": True,
            "num_calib_batches": 2,
            "percentile": 99.99
        })

        model.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(),
                    valid_loader=classification_test_dataloader())

    def test_qat_transition(self):
        model, net = self._get_trainer("test_qat_transition")
        train_params = self._get_train_params(qat_params={
            "start_epoch": 1,
            "quant_modules_calib_method": "percentile",
            "calibrate": True,
            "num_calib_batches": 2,
            "percentile": 99.99
        })

        model.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(),
                    valid_loader=classification_test_dataloader())

    def test_qat_from_calibrated_ckpt(self):
        model, net = self._get_trainer("generate_calibrated_model")
        train_params = self._get_train_params(qat_params={
            "start_epoch": 0,
            "quant_modules_calib_method": "percentile",
            "calibrate": True,
            "num_calib_batches": 2,
            "percentile": 99.99
        })

        model.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(),
                    valid_loader=classification_test_dataloader())

        calibrated_model_path = os.path.join(model.checkpoints_dir_path, "ckpt_calibrated_percentile_99.99.pth")

        model, net = self._get_trainer("test_qat_from_calibrated_ckpt")
        train_params = self._get_train_params(qat_params={
            "start_epoch": 0,
            "quant_modules_calib_method": "percentile",
            "calibrate": False,
            "calibrated_model_path": calibrated_model_path,
            "num_calib_batches": 2,
            "percentile": 99.99
        })

        model.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(),
                    valid_loader=classification_test_dataloader())


if __name__ == '__main__':
    unittest.main()
