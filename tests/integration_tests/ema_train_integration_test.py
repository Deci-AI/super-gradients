from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
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
        self.trainer = Trainer("resnet18_cifar_ema_test")
        self.model = models.get(Models.RESNET18_CIFAR, arch_params={"num_classes": 5})

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_train_exp_decay(self):
        self._init_model()
        self._train({"decay_type": "exp", "beta": 15, "decay": 0.9999})

    def test_train_threshold_decay(self):
        self._init_model()
        self._train({"decay_type": "threshold", "decay": 0.9999})

    def test_train_constant_decay(self):
        self._init_model()
        self._train({"decay_type": "constant", "decay": 0.9999})

    def test_train_with_old_ema_params(self):
        self._init_model()
        self._train({"decay": 0.9999, "exp_activation": True, "beta": 10})

    def _train(self, ema_params):
        training_params = {
            "max_epochs": 4,
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
            "train_metrics_list": [Accuracy(), Top5()],
            "valid_metrics_list": [Accuracy(), Top5()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
        }

        def before_test():
            self.assertEqual(self.trainer.net, self.trainer.ema_model.ema)

        def before_train_epoch():
            self.assertNotEqual(self.trainer.net, self.trainer.ema_model.ema)

        self.trainer.test = CallWrapper(self.trainer.test, check_before=before_test)
        self.trainer._train_epoch = CallWrapper(self.trainer._train_epoch, check_before=before_train_epoch)

        self.trainer.train(
            model=self.model, training_params=training_params, train_loader=classification_test_dataloader(), valid_loader=classification_test_dataloader()
        )

        self.assertIsNotNone(self.trainer.ema_model)


if __name__ == "__main__":
    unittest.main()
