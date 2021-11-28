from super_gradients import ClassificationTestDatasetInterface
from super_gradients.training import MultiGPUMode
from super_gradients.training import SgModel
from super_gradients.training.metrics import Accuracy, Top5
import unittest


def do_nothing():
    pass


class CallWrapper:

    def __init__(self, f, check_before=do_nothing):
        self.f = f
        self.check_before = check_before

    def __call__(self, *args, **kwargs):
        self.check_before()
        return self.f(*args, **kwargs)


class EMAIntegrationTest(unittest.TestCase):

    def _init_model(self) -> None:
        self.model = SgModel("resnet18_cifar_ema_test", model_checkpoints_location='local',
                             device='cpu', multi_gpu=MultiGPUMode.OFF)
        dataset_interface = ClassificationTestDatasetInterface({"batch_size": 32})
        self.model.connect_dataset_interface(dataset_interface, 8)
        self.model.build_model("resnet18_cifar")

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_train(self):
        self._init_model()
        self._train({})
        self._init_model()
        self._train({"exp_activation": False})

    def _train(self, ema_params):
        training_params = {"max_epochs": 4,
                           "lr_updates": [4],
                           "lr_mode": "step",
                           "lr_decay_factor": 0.1,
                           "lr_warmup_epochs": 0,
                           "initial_lr": 0.1,
                           "loss": "cross_entropy",
                           "optimizer": "SGD",
                           "criterion_params": {},
                           "ema": True,
                           "ema_params": ema_params,
                           "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                           "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                           "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                           "greater_metric_to_watch_is_better": True}

        def before_test():
            self.assertEqual(self.model.net, self.model.ema_model.ema)

        def before_train_epoch():
            self.assertNotEqual(self.model.net, self.model.ema_model.ema)

        self.model.test = CallWrapper(self.model.test, check_before=before_test)
        self.model._train_epoch = CallWrapper(self.model._train_epoch, check_before=before_train_epoch)

        self.model.train(training_params=training_params)

        self.assertIsNotNone(self.model.ema_model)


if __name__ == '__main__':
    unittest.main()
