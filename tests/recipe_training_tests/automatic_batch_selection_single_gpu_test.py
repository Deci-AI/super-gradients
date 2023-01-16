import unittest

import pkg_resources
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra import compose
from omegaconf import OmegaConf, open_dict

from super_gradients import Trainer
from super_gradients.training.utils.hydra_utils import normalize_path


class MyTestCase(unittest.TestCase):
    def test_something(self):
        GlobalHydra.instance().clear()
        sg_recipes_dir = pkg_resources.resource_filename("super_gradients.recipes", "")

        with initialize_config_dir(config_dir=normalize_path(sg_recipes_dir), version_base="1.2"):
            cfg = compose(config_name="cifar10_resnet")
            cfg.experiment_name = "batch_size_selection_test_no_max"
            cfg.training_hyperparams.max_epochs = 1
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.pre_launch_callbacks_list = [
                    OmegaConf.create({"AutoTrainBatchSizeSelectionCallback": {"min_batch_size": 64, "size_step": 10000, "num_forward_passes": 3}})
                ]
        Trainer.train_from_config(cfg)


if __name__ == "__main__":
    unittest.main()
