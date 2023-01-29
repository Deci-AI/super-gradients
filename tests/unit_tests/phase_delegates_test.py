import unittest

from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models import LeNet
from super_gradients.training.utils.callbacks import Phase, PhaseCallback, PhaseContext


class ContextMethodsCheckerCallback(PhaseCallback):
    """
    Callback for checking that at a certain phase specific Trainer methods are accessible.
    """

    def __init__(self, phase: Phase, accessible_method_names: list, non_accessible_method_names: list):
        super(ContextMethodsCheckerCallback, self).__init__(phase)
        self.accessible_method_names = accessible_method_names
        self.non_accessible_method_names = non_accessible_method_names
        self.result = True

    def __call__(self, context: PhaseContext):
        for accessible_method_name in self.accessible_method_names:
            if not hasattr(context.context_methods, accessible_method_name):
                self.result = False

        for non_accessible_method_name in self.non_accessible_method_names:
            if hasattr(context.context_methods, non_accessible_method_name):
                self.result = False


class ContextMethodsTest(unittest.TestCase):
    def test_access_to_methods_by_phase(self):
        net = LeNet()
        trainer = Trainer("test_access_to_methods_by_phase")

        phase_callbacks = []
        for phase in Phase:
            if phase in [
                Phase.PRE_TRAINING,
                Phase.TRAIN_EPOCH_START,
                Phase.TRAIN_EPOCH_END,
                Phase.VALIDATION_EPOCH_END,
                Phase.VALIDATION_END_BEST_EPOCH,
                Phase.POST_TRAINING,
            ]:
                phase_callbacks.append(
                    ContextMethodsCheckerCallback(
                        phase=phase,
                        accessible_method_names=["get_net", "set_net", "set_ckpt_best_name", "reset_best_metric", "validate_epoch"],
                        non_accessible_method_names=[],
                    )
                )
            else:
                phase_callbacks.append(
                    ContextMethodsCheckerCallback(
                        phase=phase,
                        non_accessible_method_names=["get_net", "set_net", "set_ckpt_best_name", "reset_best_metric", "validate_epoch", "set_ema"],
                        accessible_method_names=[],
                    )
                )

        train_params = {
            "max_epochs": 1,
            "lr_updates": [],
            "lr_decay_factor": 0.1,
            "lr_mode": "step",
            "lr_warmup_epochs": 0,
            "initial_lr": 1,
            "loss": "cross_entropy",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": phase_callbacks,
        }

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4),
            valid_loader=classification_test_dataloader(batch_size=4),
        )
        for phase_callback in phase_callbacks:
            if isinstance(phase_callback, ContextMethodsCheckerCallback):
                self.assertTrue(phase_callback.result)


if __name__ == "__main__":
    unittest.main()
