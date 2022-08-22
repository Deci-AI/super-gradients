import unittest
from super_gradients.training import Trainer
from super_gradients.training.metrics import Accuracy
from super_gradients.training.datasets import ClassificationTestDatasetInterface
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
import torch


class TestInputSizesCallback(PhaseCallback):
    """
    Phase callback that collects the input shapes rates in lr_placeholder at the end of each forward pass.
    """

    def __init__(self, shapes_placeholder):
        super(TestInputSizesCallback, self).__init__(Phase.TRAIN_BATCH_END)
        self.shapes_placeholder = shapes_placeholder

    def __call__(self, context: PhaseContext):
        self.shapes_placeholder.append(context.inputs.shape)


def test_forward_pass_prep_fn(inputs, targets, *args, **kwargs):
    inputs = torch.nn.functional.interpolate(
        inputs, size=(50, 50), mode="bilinear", align_corners=False
    )
    return inputs, targets


class ForwardpassPrepFNTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_params = {"batch_size": 4}
        self.dataset = ClassificationTestDatasetInterface(dataset_params=self.dataset_params)
        self.arch_params = {'num_classes': 10}

    def test_resizing_with_forward_pass_prep_fn(self):
        # Define Model
        trainer = Trainer("ForwardpassPrepFNTest")
        trainer.connect_dataset_interface(self.dataset)
        trainer.build_model("resnet18", arch_params=self.arch_params)

        sizes = []
        phase_callbacks = [TestInputSizesCallback(sizes)]

        train_params = {"max_epochs": 2, "cosine_final_lr_ratio": 0.2, "lr_mode": "cosine",
                        "lr_cooldown_epochs": 2,
                        "lr_warmup_epochs": 3, "initial_lr": 1, "loss": "cross_entropy", "optimizer": 'SGD',
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True, "ema": False, "phase_callbacks": phase_callbacks,
                        "pre_prediction_callback": test_forward_pass_prep_fn}
        trainer.train(train_params)

        # ALTHOUGH NOT SEEN IN HERE, THE 4TH EPOCH USES LR=1, SO THIS IS THE EXPECTED LIST AS WE COLLECT
        # THE LRS AFTER THE UPDATE
        sizes = list(map(lambda size: size[2], sizes))
        self.assertTrue(all(map(lambda size: size == 50, sizes)))
