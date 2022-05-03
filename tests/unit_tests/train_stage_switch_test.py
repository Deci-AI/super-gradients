import unittest
from super_gradients.training import SgModel
from super_gradients.training.metrics import Accuracy
from super_gradients.training.datasets import ClassificationTestDatasetInterface
from super_gradients.training.models import LeNet
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext, TrainingStageSwitchCallbackBase


class CriterionReductionAttributeCollectorCallback(PhaseCallback):
    """
    Phase callback that collects the 'reduction' attribute from context.criterion to placeholder.
    """

    def __init__(self, placeholder: list):
        super(CriterionReductionAttributeCollectorCallback, self).__init__(Phase.TRAIN_EPOCH_END)
        self.placeholder = placeholder

    def __call__(self, context: PhaseContext):
        self.placeholder.append(context.criterion.reduction)


class TestTrainingStageSwitchCallback(TrainingStageSwitchCallbackBase):
    def apply_stage_change(self, context: PhaseContext):
        context.criterion.reduction = 'sum'


class TrainStageSwitchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_params = {"batch_size": 4}
        self.dataset = ClassificationTestDatasetInterface(dataset_params=self.dataset_params)
        self.arch_params = {'num_classes': 10}

    def test_train_stage_switch(self):
        # Define Model
        net = LeNet()
        model = SgModel("stage_switch_test", model_checkpoints_location='local')
        model.connect_dataset_interface(self.dataset)
        model.build_model(net, arch_params=self.arch_params)

        criterion_reductions = []
        phase_callbacks = [TestTrainingStageSwitchCallback(2),
                           CriterionReductionAttributeCollectorCallback(placeholder=criterion_reductions)]

        train_params = {"max_epochs": 4, "lr_updates": [], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 1, "loss": "cross_entropy", "optimizer": 'SGD',
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Accuracy()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True, "ema": False, "phase_callbacks": phase_callbacks}

        expected_reductions = ['mean', 'mean', 'sum', 'sum']
        model.train(train_params)
        self.assertListEqual(criterion_reductions, expected_reductions)


if __name__ == '__main__':
    unittest.main()
