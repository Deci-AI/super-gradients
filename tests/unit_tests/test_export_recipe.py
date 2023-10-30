import tempfile
import unittest
import os

import hydra
from hydra import initialize_config_dir, compose

from super_gradients.common.environment.cfg_utils import export_recipe


class TestExportRecipe(unittest.TestCase):
    def test_export_recipe(self):
        with tempfile.TemporaryDirectory() as td:
            save_path = os.path.join(td, "cifar10_resnet_complete.yaml")
            # Define the command to run your script
            export_recipe(config_name="cifar10_resnet", save_path=save_path)

            # Check if the output file was created
            self.assertTrue(os.path.exists(save_path))

            with initialize_config_dir(config_dir=td, version_base="1.2"):
                cfg = compose(config_name="cifar10_resnet_complete.yaml")

            cfg = hydra.utils.instantiate(cfg)

            self.assertEqual(cfg.training_hyperparams.max_epochs, 250)


if __name__ == "__main__":
    unittest.main()
