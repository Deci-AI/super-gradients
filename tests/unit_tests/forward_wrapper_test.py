import unittest

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models import LeNet
from super_gradients.training.utils.callbacks import PhaseContext, Callback
import torch


class OutputsCollectorCallback(Callback):
    def __init__(self):
        self.validation_outputs = []
        self.train_outputs = []

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        self.validation_outputs.append(context.preds)

    def on_train_batch_end(self, context: PhaseContext) -> None:
        self.train_outputs.append(context.preds)


class DummyForwardWrapper:
    def __call__(self, inputs: torch.Tensor, model: torch.nn.Module):
        return torch.ones_like(model(inputs))


def compare_tensor_lists(list1, list2):
    if len(list1) != len(list2):
        return False

    # Move tensors to CPU
    list1 = [t.cpu() for t in list1]
    list2 = [t.cpu() for t in list2]

    for tensor1, tensor2 in zip(list1, list2):
        if not torch.all(torch.eq(tensor1, tensor2)):
            return False
    return True


class TestForwardWrapper(unittest.TestCase):
    def test_train_with_validation_forward_wrapper(self):
        # Define Model
        net = LeNet()
        trainer = Trainer("test_train_with_validation_forward_wrapper")
        output_collector = OutputsCollectorCallback()
        validation_forward_wrapper = DummyForwardWrapper()
        train_params = {
            "max_epochs": 1,
            "initial_lr": 0.1,
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": [output_collector],
            "warmup_mode": "LinearEpochLRWarmup",
            "validation_forward_wrapper": validation_forward_wrapper,
            "average_best_models": False,
        }

        expected_outputs = [torch.ones(4, 10)]
        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4),
            valid_loader=classification_test_dataloader(batch_size=4),
        )
        self.assertTrue(compare_tensor_lists(expected_outputs, output_collector.validation_outputs))
        self.assertFalse(compare_tensor_lists(expected_outputs, output_collector.train_outputs))


if __name__ == "__main__":
    unittest.main()
