import shutil
import unittest
import os

import torch
from super_gradients.common.environment import environment_config


class Cifar10RecipeSanityUnitTest(unittest.TestCase):
    def test_cifar10_resnet_metric(self):
        ckpt_dir = os.path.join(environment_config.PKG_CHECKPOINTS_DIR, "cifar10_resnet_sanity")
        sd = torch.load(os.path.join(ckpt_dir, "ckpt_best.pth"))
        shutil.rmtree(ckpt_dir)
        self.assertTrue(sd["acc"].cpu().item() >= 0.75)


if __name__ == "__main__":
    unittest.main()
