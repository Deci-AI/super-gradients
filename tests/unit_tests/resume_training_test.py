import os
import unittest
from copy import deepcopy

from torchmetrics import Metric

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.models import LeNet
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path, get_latest_run_id
import torch


class FirstEpochInfoCollector(PhaseCallback):
    def __init__(self):
        super().__init__(phase=Phase.TRAIN_EPOCH_START)
        self.called = False
        self.first_epoch = None
        self.first_epoch_net = None

    def __call__(self, context: PhaseContext):
        if not self.called:
            self.first_epoch = context.epoch
            self.first_epoch_net = deepcopy(context.net)
            self.called = True


class DummyEpochMetric(Metric):
    """
    Dummy metric that returns 10 if epoch is smaller then 3 else 2
    """

    def __init__(self):
        super(DummyEpochMetric, self).__init__()
        self.add_state("curr_epoch", default=torch.tensor(0.0))

    def update(self, epoch: int):
        self.curr_epoch = torch.tensor(epoch)

    def compute(self) -> torch.Tensor:
        if self.curr_epoch < 3:
            return torch.tensor(10)
        else:
            return torch.tensor(2)


class DummyMetric1(Metric):
    """
    Dummy metric that always returns 1
    """

    def __init__(self):
        super(DummyMetric1, self).__init__()
        self.add_state("curr_val", default=torch.tensor(0.0))

    def update(self, epoch: int):
        self.curr_val = torch.tensor(1.0)

    def compute(self) -> torch.Tensor:
        return self.curr_val


class ResumeTrainingTest(unittest.TestCase):
    def test_resume_training(self):
        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "StepLRScheduler",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }

        # Define Model
        net = LeNet()
        trainer = Trainer("test_resume_training")
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 3 AND THE END OF EPOCH NUMBER 2
        resume_net = LeNet()
        trainer = Trainer("test_resume_training")
        first_epoch_cb = FirstEpochInfoCollector()
        train_params["resume"] = True
        train_params["max_epochs"] = 3
        train_params["phase_callbacks"] = [first_epoch_cb]
        trainer.train(
            model=resume_net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        # ASSERT RELOADED MODEL HAS THE SAME WEIGHTS AS THE MODEL SAVED IN FIRST PART OF TRAINING
        self.assertTrue(check_models_have_same_weights(net, first_epoch_cb.first_epoch_net))

        # ASSERT WE START FROM THE RIGHT EPOCH NUMBER
        self.assertTrue(first_epoch_cb.first_epoch == 2)

    def test_resume_run_id_training(self):
        ckpt_root_dir = ""
        experiment_name = "test_resume_training"

        experiment_dir = get_checkpoints_dir_path(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name)
        original_dir_count = len(os.listdir(experiment_dir))

        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "StepLRScheduler",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }

        # FIRST TRAINING - Train for 1 epoch
        net_v1 = LeNet()
        trainer = Trainer(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name)
        trainer.train(model=net_v1, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())
        first_run_id = get_latest_run_id(checkpoints_root_dir=ckpt_root_dir, experiment_name=experiment_name)

        # Check directory size
        self.assertEqual(original_dir_count + 1, len(os.listdir(experiment_dir)), "You should have 1 run folder created only after calling `Trainer.train`.")

        # SECOND TRAINING - Train for 1 epoch
        net_v2 = LeNet()  # We don't want to override the first model
        trainer = Trainer(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name)
        trainer.train(model=net_v2, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())
        second_run_id = get_latest_run_id(checkpoints_root_dir=ckpt_root_dir, experiment_name=experiment_name)

        # Check directory size
        self.assertEqual(
            original_dir_count + 2, len(os.listdir(experiment_dir)), "You should have 2 run folder created only after calling `Trainer.train` twice."
        )
        self.assertNotEqual(first_run_id, second_run_id, "First and Second trainings should have different run ids.")

        # RESUME
        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 3 AND THE END OF EPOCH NUMBER 2
        first_epoch_cb = FirstEpochInfoCollector()
        train_params["run_id"] = first_run_id  # Let's run on the first run and make sure it works great
        train_params["max_epochs"] = 3
        train_params["phase_callbacks"] = [first_epoch_cb]

        trainer = Trainer(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name)
        trainer.train(
            model=LeNet(),
            training_params=train_params,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(),
        )

        self.assertTrue(check_models_have_same_weights(net_v1, first_epoch_cb.first_epoch_net))
        self.assertFalse(check_models_have_same_weights(net_v2, first_epoch_cb.first_epoch_net))
        self.assertTrue(first_epoch_cb.first_epoch == 2)

        # Resuming should not create a new run
        self.assertEqual(
            original_dir_count + 2,
            len(os.listdir(experiment_dir)),
            "You should have only 2 run folder created only after calling `Trainer.train` twice and resuming it once.",
        )

    def test_resume_external_training(self):
        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "StepLRScheduler",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }
        # Define Model
        net = LeNet()
        trainer = Trainer("test_resume_training")
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 3 AND THE END OF EPOCH NUMBER 2
        resume_net = LeNet()
        resume_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_latest.pth")

        # SET DIFFERENT EXPERIMENT NAME SO WE LOAD A CHECKPOINT THAT HAS A DIFFERENT PATH FROM THE DEFAULT ONE
        trainer = Trainer("test_resume_external_training")
        first_epoch_cb = FirstEpochInfoCollector()
        train_params["resume_path"] = resume_path
        train_params["max_epochs"] = 3
        train_params["phase_callbacks"] = [first_epoch_cb]
        trainer.train(
            model=resume_net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        # ASSERT RELOADED MODEL HAS THE SAME WEIGHTS AS THE MODEL SAVED IN FIRST PART OF TRAINING
        self.assertTrue(check_models_have_same_weights(net, first_epoch_cb.first_epoch_net))

        # ASSERT WE START FROM THE RIGHT EPOCH NUMBER
        self.assertTrue(first_epoch_cb.first_epoch == 2)

    def test_resume_external_training_same_dir(self):
        ckpt_root_dir = ""
        experiment_name = "test_resume_training"

        experiment_dir = get_checkpoints_dir_path(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name)
        original_dir_count = len(os.listdir(experiment_dir))

        train_params = {
            "max_epochs": 2,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "StepLRScheduler",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }

        # Train for 1 more epoch
        net = LeNet()
        trainer = Trainer(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name)
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        # Check directory size
        self.assertEqual(original_dir_count + 1, len(os.listdir(experiment_dir)), "You should have 1 run folder created only after calling `Trainer.train`.")

        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 3 AND THE END OF EPOCH NUMBER 2
        resume_net = LeNet()
        resume_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_latest.pth")

        # SET DIFFERENT EXPERIMENT NAME SO WE LOAD A CHECKPOINT THAT HAS A DIFFERENT PATH FROM THE DEFAULT ONE
        trainer = Trainer(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_dir)
        first_epoch_cb = FirstEpochInfoCollector()
        train_params["resume_path"] = resume_path
        train_params["max_epochs"] = 3
        train_params["phase_callbacks"] = [first_epoch_cb]
        trainer.train(
            model=resume_net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        # ASSERT RELOADED MODEL HAS THE SAME WEIGHTS AS THE MODEL SAVED IN FIRST PART OF TRAINING
        self.assertTrue(check_models_have_same_weights(net, first_epoch_cb.first_epoch_net))

        # ASSERT WE START FROM THE RIGHT EPOCH NUMBER
        self.assertTrue(first_epoch_cb.first_epoch == 2)

        # Resuming should create a new run
        self.assertEqual(
            original_dir_count + 2,
            len(os.listdir(experiment_dir)),
            "Using resume_path should create a new run folder",
        )

    def test_resume_training_different_metric_to_watch(self):
        """
        Tests that if we switch metrics when returning the best_metric
         is properly extracted by performing additional test.
         We use the dummy epoch metric that will not be optimal in the latest checkpoint on purpose - to check we are
          not just using the latest metric.
        """
        train_params = {
            "max_epochs": 4,
            "lr_updates": [1],
            "lr_decay_factor": 0.1,
            "lr_mode": "StepLRScheduler",
            "lr_warmup_epochs": 0,
            "initial_lr": 0.1,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [],
            "valid_metrics_list": [DummyEpochMetric()],
            "metric_to_watch": "DummyEpochMetric",
            "greater_metric_to_watch_is_better": True,
            "average_best_models": False,
        }

        # Define Model
        net = LeNet()
        trainer = Trainer("test_resume_training")
        trainer.train(model=net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader())

        # BEST METRIC WILL BE 3 SINCE AT EPOCH 4 IT WILL BE 10 (THIS IS DONE TO CHECK WE ARE NOT TAKING JUST THE LATEST)
        self.assertEqual(trainer.best_metric, 10)

        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 4 AND THE END OF EPOCH NUMBER 3
        resume_net = LeNet()
        trainer = Trainer("test_resume_training")
        train_params["resume"] = True
        train_params["max_epochs"] = 5

        # CHANGE THE METRIC AND METRIC TO WATCH
        train_params["valid_metrics_list"] = [DummyMetric1()]
        train_params["metric_to_watch"] = "DummyMetric1"
        first_epoch_cb = FirstEpochInfoCollector()
        train_params["phase_callbacks"] = [first_epoch_cb]
        trainer.train(
            model=resume_net, training_params=train_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        # ASSERT RELOADED MODEL HAS THE SAME WEIGHTS AS THE MODEL SAVED IN FIRST PART OF TRAINING
        self.assertTrue(check_models_have_same_weights(net, first_epoch_cb.first_epoch_net))

        # ASSERT WE START FROM THE RIGHT EPOCH NUMBER
        self.assertTrue(first_epoch_cb.first_epoch == 4)

        # EVEN THOUGH BEST METRIC IS BEFORE RESUME WAS 2 WE ARE SWITCHING METRICS SO THE BEST SHOULD BE 1
        self.assertTrue(trainer.best_metric, 1)


if __name__ == "__main__":
    unittest.main()
