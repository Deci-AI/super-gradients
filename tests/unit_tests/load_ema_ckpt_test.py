import unittest
from super_gradients.training import SgModel
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils.utils import check_models_have_same_weights
from super_gradients.training.datasets import ClassificationTestDatasetInterface
from super_gradients.training.models import LeNet


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
        model.build_model(net, arch_params={'num_classes': 10})

        model.train(self.train_params)

        ema_model = model.ema_model.ema

        net = LeNet()
        model = SgModel("ema_ckpt_test", model_checkpoints_location='local')
        model.build_model(net, arch_params={'num_classes': 10, 'load_checkpoint': True})
        model.connect_dataset_interface(self.dataset)

        # TRAIN FOR 0 EPOCHS JUST TO SEE THAT WHEN CONTINUING TRAINING EMA MODEL HAS BEEN SAVED CORRECTLY
        model.train(self.train_params)

        reloaded_ema_model = model.ema_model.ema

        # ASSERT RELOADED EMA MODEL HAS THE SAME WEIGHTS AS THE EMA MODEL SAVED IN FIRST PART OF TRAINING
        assert check_models_have_same_weights(ema_model, reloaded_ema_model)


if __name__ == '__main__':
    unittest.main()
