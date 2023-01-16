import unittest
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from super_gradients.training.models import LeNet


class LastBatchIdxCollector(PhaseCallback):
    def __init__(self, train: bool = True):
        phase = Phase.TRAIN_BATCH_END if train else Phase.VALIDATION_BATCH_END
        super().__init__(phase=phase)
        self.last_batch_idx = 0

    def __call__(self, context: PhaseContext):
        self.last_batch_idx = context.batch_idx


class MaxBatchesLoopBreakTest(unittest.TestCase):
    def test_max_train_batches_loop_break(self):
        last_batch_collector = LastBatchIdxCollector()
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
            "phase_callbacks": [last_batch_collector],
            "max_train_batches": 3,
        }

        # Define Model
        net = LeNet()
        trainer = Trainer("test_max_batches_break_train")
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(dataset_size=16, batch_size=4),
            valid_loader=classification_test_dataloader(),
        )

        # ASSERT LAST BATCH IDX IS 2
        print(last_batch_collector.last_batch_idx)
        self.assertTrue(last_batch_collector.last_batch_idx == 2)

    def test_max_valid_batches_loop_break(self):
        last_batch_collector = LastBatchIdxCollector(train=False)
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
            "phase_callbacks": [last_batch_collector],
            "max_valid_batches": 3,
        }

        # Define Model
        net = LeNet()
        trainer = Trainer("test_max_batches_break_val")
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(),
            valid_loader=classification_test_dataloader(dataset_size=16, batch_size=4),
        )

        # ASSERT LAST BATCH IDX IS 2
        self.assertTrue(last_batch_collector.last_batch_idx == 2)


if __name__ == "__main__":
    unittest.main()
