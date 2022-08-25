import shutil
import unittest

from super_gradients.training import models

import super_gradients
import torch
import os
from super_gradients import Trainer, ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy, Top5


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super_gradients.init_trainer()
        # NAMES FOR THE EXPERIMENTS TO LATER DELETE
        cls.folder_names = ['test_train', 'test_save_load', 'test_load_w', 'test_load_w2',
                            'test_load_w3', 'test_checkpoint_content', 'analyze']
        cls.training_params = {"max_epochs": 1,
                               "silent_mode": True,
                               "lr_decay_factor": 0.1,
                               "initial_lr": 0.1,
                               "lr_updates": [4],
                               "lr_mode": "step",
                               "loss": "cross_entropy", "train_metrics_list": [Accuracy(), Top5()],
                               "valid_metrics_list": [Accuracy(), Top5()],
                               "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                               "greater_metric_to_watch_is_better": True}

    @classmethod
    def tearDownClass(cls) -> None:
        # ERASE ALL THE FOLDERS THAT WERE CREATED DURING THIS TEST
        for folder in cls.folder_names:
            if os.path.isdir(os.path.join('checkpoints', folder)):
                shutil.rmtree(os.path.join('checkpoints', folder))

    @staticmethod
    def get_classification_trainer(name=''):
        trainer = Trainer(name, model_checkpoints_location='local')
        dataset_params = {"batch_size": 4}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params, image_size=224)
        trainer.connect_dataset_interface(dataset)
        model = models.get("resnet18", arch_params={"num_classes": 5})
        return trainer, model

    def test_train(self):
        trainer, model = self.get_classification_trainer(self.folder_names[0])
        trainer.train(model=model, training_params=self.training_params)

    def test_save_load(self):
        trainer, model = self.get_classification_trainer(self.folder_names[1])
        trainer.train(model=model, training_params=self.training_params)

        resume_training_params = self.training_params.copy()
        resume_training_params["resume"] = True
        resume_training_params["max_epochs"] = 2
        trainer, model = self.get_classification_trainer(self.folder_names[1])
        trainer.train(model=model, training_params=resume_training_params)

    def test_checkpoint_content(self):
        """VERIFY THAT ALL CHECKPOINTS ARE SAVED AND CONTAIN ALL THE EXPECTED KEYS"""
        trainer, model = self.get_classification_trainer(self.folder_names[5])
        params = self.training_params.copy()
        params["save_ckpt_epoch_list"] = [1]
        trainer.train(model=model, training_params=params)
        ckpt_filename = ['ckpt_best.pth', 'ckpt_latest.pth', 'ckpt_epoch_1.pth']
        ckpt_paths = [os.path.join(trainer.checkpoints_dir_path, suf) for suf in ckpt_filename]
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path)
            self.assertListEqual(['net', 'acc', 'epoch', 'optimizer_state_dict', 'scaler_state_dict'],
                                 list(ckpt.keys()))
        trainer._save_checkpoint()
        weights_only = torch.load(os.path.join(trainer.checkpoints_dir_path, 'ckpt_latest_weights_only.pth'))
        self.assertListEqual(['net'], list(weights_only.keys()))


if __name__ == '__main__':
    unittest.main()
