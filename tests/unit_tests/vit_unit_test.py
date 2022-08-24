import unittest
from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationTestDatasetInterface
from super_gradients import Trainer
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training import models


class TestViT(unittest.TestCase):

    def setUp(self):
        self.arch_params = HpmStruct(**{"image_size": (224, 224), "patch_size": (16, 16), "num_classes": 10})
        self.dataset = ClassificationTestDatasetInterface(dataset_params={"batch_size": 16})

        self.train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                             "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                             "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                             "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                             "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy"}

    def test_train_vit(self):
        """
        Validate vit_base
        """
        trainer = Trainer("test_vit_base", device='cpu')
        trainer.connect_dataset_interface(self.dataset, data_loader_num_workers=8)
        model = models.get('vit_base', arch_params={"num_classes": 5})
        trainer.train(model=model, training_params=self.train_params)


if __name__ == '__main__':
    unittest.main()
