import unittest
from super_gradients import Trainer
from super_gradients.common.auto_logging.auto_logger import AutoLoggerConfig
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18
import os
import logging
from super_gradients.common.abstractions.abstract_logger import get_logger
import shutil


class SgTrainerLoggingTest(unittest.TestCase):
    def test_train_logging(self):
        trainer = Trainer("test_train_with_full_log")

        net = ResNet18(num_classes=5, arch_params={})
        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=10),
            valid_loader=classification_test_dataloader(batch_size=10),
        )

        logfile_path = AutoLoggerConfig.get_log_file_path()
        assert os.path.exists(logfile_path) and os.path.getsize(logfile_path) > 0

        root_logger_handlers = logging.root.handlers
        assert any(isinstance(handler, logging.handlers.FileHandler) and handler.baseFilename == logfile_path for handler in root_logger_handlers)
        assert any(isinstance(handler, logging.StreamHandler) and handler.name == "console" for handler in root_logger_handlers)

    def test_logger_with_non_existing_deci_logs_dir(self):
        user_dir = os.path.expanduser(r"~")
        logs_dir_path = os.path.join(user_dir, "non_existing_deci_logs_dir")
        if os.path.exists(logs_dir_path):
            shutil.rmtree(logs_dir_path)

        module_name = "super_gradients.trainer.sg_trainer"

        _ = get_logger(module_name, logs_dir_path=logs_dir_path)
        root_logger_handlers = logging.root.handlers
        assert any(isinstance(handler, logging.StreamHandler) and handler.name == "console" for handler in root_logger_handlers)


if __name__ == "__main__":
    unittest.main()
