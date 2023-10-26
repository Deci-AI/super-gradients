import unittest
import tempfile

from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18
from super_gradients.training.utils.distributed_training_utils import setup_device


class TestMixedPrecisionDisabled(unittest.TestCase):
    def test_mixed_precision_automatically_changed_with_warning(self):
        setup_device(device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = Trainer("test_mixed_precision_automatically_changed_with_warning", ckpt_root_dir=temp_dir)
            net = ResNet18(num_classes=5, arch_params={})
            train_params = {
                "max_epochs": 2,
                "lr_updates": [1],
                "lr_decay_factor": 0.1,
                "lr_mode": "StepLRScheduler",
                "lr_warmup_epochs": 0,
                "initial_lr": 0.1,
                "loss": "CrossEntropyLoss",
                "criterion_params": {"ignore_index": 0},
                "train_metrics_list": [Accuracy(), Top5()],
                "valid_metrics_list": [Accuracy(), Top5()],
                "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True,
                "mixed_precision": True,  # This is not supported for CPU, so we expect a warning to be raised AND the code to run
            }
            import warnings

            with warnings.catch_warnings(record=True) as w:
                # Trigger a filter to always make warnings visible
                warnings.simplefilter("always")

                trainer.train(
                    model=net,
                    training_params=train_params,
                    train_loader=classification_test_dataloader(batch_size=10),
                    valid_loader=classification_test_dataloader(batch_size=10),
                )

                # Check if the desired warning is in the list of warnings
                self.assertTrue(any("Mixed precision training is not supported on CPU" in str(warn.message) for warn in w))


if __name__ == "__main__":
    unittest.main()
