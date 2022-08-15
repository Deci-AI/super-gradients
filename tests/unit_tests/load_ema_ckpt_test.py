import unittest
from copy import deepcopy

from super_gradients.training import SgModel
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.datasets import ClassificationTestDatasetInterface
from super_gradients.training.models import LeNet


class PreTrainingEMANetCollector(PhaseCallback):
    def __init__(self):
        super(PreTrainingEMANetCollector, self).__init__(phase=Phase.PRE_TRAINING)
        self.net = None

    def __call__(self, context: PhaseContext):
        self.net = deepcopy(context.ema_model)


class LoadCheckpointWithEmaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_params = {"batch_size": 4}
        self.dataset = ClassificationTestDatasetInterface(dataset_params=self.dataset_params)
        self.train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                             "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": 'SGD',
                             "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                             "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                             "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                             "greater_metric_to_watch_is_better": True, "ema": True}

    def test_ema_ckpt_reload(self):
        # Define Model
        net = LeNet()
        model = SgModel("ema_ckpt_test", model_checkpoints_location='local')

        model.connect_dataset_interface(self.dataset)

        model.train(net=net, training_params=self.train_params)

        ema_model = model.ema_model.ema

        # TRAIN FOR 1 MORE EPOCH AND COMPARE THE NET AT THE BEGINNING OF EPOCH 3 AND THE END OF EPOCH NUMBER 2
        net = LeNet()
        model = SgModel("ema_ckpt_test", model_checkpoints_location='local')
        model.connect_dataset_interface(self.dataset)
        net_collector = PreTrainingEMANetCollector()
        self.train_params["resume"] = True
        self.train_params["max_epochs"] = 3
        self.train_params["phase_callbacks"] = [net_collector]
        model.train(net=net, training_params=self.train_params)

        reloaded_ema_model = net_collector.net.ema

        # ASSERT RELOADED EMA MODEL HAS THE SAME WEIGHTS AS THE EMA MODEL SAVED IN FIRST PART OF TRAINING
        assert check_models_have_same_weights(ema_model, reloaded_ema_model)


if __name__ == '__main__':
    unittest.main()
