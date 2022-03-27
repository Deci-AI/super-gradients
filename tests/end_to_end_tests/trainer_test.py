import shutil
import unittest

import super_gradients
import torch
import os
from super_gradients import SgModel, ClassificationTestDatasetInterface
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
        model = SgModel(name, model_checkpoints_location='local')
        dataset_params = {"batch_size": 4}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)
        model.build_model("resnet18_cifar")
        return model

    def test_train(self):
        model = self.get_classification_trainer(self.folder_names[0])
        model.train(training_params=self.training_params)

    def test_save_load(self):
        model = self.get_classification_trainer(self.folder_names[1])
        model.train(training_params=self.training_params)
        model.build_model("resnet18_cifar", checkpoint_params={'load_checkpoint': True})

    def test_load_only_weights_from_ckpt(self):
        # Create a checkpoint with 100% accuracy
        model = self.get_classification_trainer(self.folder_names[2])
        params = self.training_params.copy()

        params['max_epochs'] = 3
        model.train(training_params=params)
        # Build a model that continues the training
        model = self.get_classification_trainer(self.folder_names[3])
        model.build_model('resnet18_cifar', checkpoint_params={"load_checkpoint": True, "load_weights_only": False,
                                                         "source_ckpt_folder_name": self.folder_names[2]}
                          )
        self.assertTrue(model.best_metric > -1)
        self.assertTrue(model.start_epoch != 0)
        # start_epoch is not initialized, adding to max_epochs
        self.training_params['max_epochs'] += 3
        model.train(training_params=self.training_params)
        # Build a model that loads the weights and starts from scratch
        model = self.get_classification_trainer(self.folder_names[4])
        model.build_model('resnet18_cifar', checkpoint_params={"load_checkpoint": True, "load_weights_only": True,
                                                         "source_ckpt_folder_name": self.folder_names[2]}
                          )
        self.assertTrue(model.best_metric == -1)
        self.assertTrue(model.start_epoch == 0)
        self.training_params['max_epochs'] += 3
        model.train(training_params=self.training_params)

    def test_checkpoint_content(self):
        """VERIFY THAT ALL CHECKPOINTS ARE SAVED AND CONTAIN ALL THE EXPECTED KEYS"""
        model = self.get_classification_trainer(self.folder_names[5])
        params = self.training_params.copy()
        params["save_ckpt_epoch_list"] = [1]
        model.train(training_params=params)
        ckpt_filename = ['ckpt_best.pth', 'ckpt_latest.pth', 'ckpt_epoch_1.pth']
        ckpt_paths = [os.path.join(model.checkpoints_dir_path, suf) for suf in ckpt_filename]
        for ckpt_path in ckpt_paths:
            ckpt = torch.load(ckpt_path)
            self.assertListEqual(['net', 'acc', 'epoch', 'optimizer_state_dict', 'scaler_state_dict'],
                                 list(ckpt.keys()))
        model.save_checkpoint()
        weights_only = torch.load(os.path.join(model.checkpoints_dir_path, 'ckpt_latest_weights_only.pth'))
        self.assertListEqual(['net'], list(weights_only.keys()))

    def test_compute_model_runtime(self):
        model = self.get_classification_trainer(self.folder_names[6])
        model.compute_model_runtime()
        model.compute_model_runtime(batch_sizes=1, input_dims=(3, 224, 224), verbose=False)
        model.compute_model_runtime(batch_sizes=[1, 2, 3], verbose=True)
        # VERIFY MODEL RETURNS TO PREVIOUS TRAINING MODE
        model.net.train()
        model.compute_model_runtime(batch_sizes=1, verbose=False)
        assert model.net.training, 'MODEL WAS SET TO eval DURING compute_model_runtime, BUT DIDN\'t RETURN TO PREVIOUS'
        model.net.eval()
        model.compute_model_runtime(batch_sizes=1, verbose=False)
        assert not model.net.training, 'MODEL WAS SET TO eval DURING compute_model_runtime, BUT RETURNED TO TRAINING'

        # THESE SHOULD HANDLE THE EXCEPTION OF CUDA OUT OF MEMORY
        if torch.cuda.is_available():
            model._switch_device('cuda')
            model.compute_model_runtime(batch_sizes=10000, verbose=False, input_dims=(3, 224, 224))
            model.compute_model_runtime(batch_sizes=[10000, 10, 50, 100, 1000, 5000], verbose=True)

    def test_predict(self):
        model = self.get_classification_trainer(self.folder_names[6])
        inputs = torch.randn((5, 3, 32, 32))
        targets = torch.randint(0, 5, (5, 1))
        model.predict(inputs=inputs, targets=targets)
        model.predict(inputs=inputs, targets=targets, half=True)
        model.predict(inputs=inputs, targets=targets, half=False, verbose=True)


if __name__ == '__main__':
    unittest.main()
