import unittest

import hydra
import pkg_resources

import super_gradients
from super_gradients import ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy


class YamlLoadingTest(unittest.TestCase):

    def setUp(self):
        super_gradients.init_trainer()

    def test_training_from_yaml(self):
        self.train_model()

    @staticmethod
    @hydra.main(config_path='super_gradients/recipes', config_name='test_resnet.yaml')
    def train_model(cfg):

        # INSTANTIATE ALL OBJECTS IN CFG
        cfg.training_hyperparams['max_epochs'] = 1

        cfg = hydra.utils.instantiate(cfg)

        # CONNECT THE DATASET INTERFACE WITH DECI MODEL
        cfg.sg_model.connect_dataset_interface(cfg.dataset_interface, data_loader_num_workers=cfg.data_loader_num_workers)

        # BUILD NETWORK
        cfg.sg_model.build_model(cfg.architecture, arch_params=cfg.arch_params, load_checkpoint=cfg.load_checkpoint)
        cfg.sg_model.train(training_params=cfg.training_hyperparams)
        sg_model = cfg.sg_model

        # WE CAN'T ASSERT USING THE UNITEST ASSERTIONS SINCE HYDRA.MAIN DOES NOT ALLOW THIS
        # FUNCTION TO HAVE ANY PARAMETERS OR RETURN VALUES
        assert isinstance(sg_model.train_metrics.Accuracy, Accuracy)
        assert isinstance(sg_model.dataset_interface, ClassificationTestDatasetInterface)
        assert sg_model.dataset_interface.dataset_params.batch_size == 10


if __name__ == '__main__':
    unittest.main()
