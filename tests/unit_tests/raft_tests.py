import unittest

import torch

from super_gradients.common.object_names import Models
from super_gradients.training import models


class TestRAFT(unittest.TestCase):
    def setUp(self):
        self.models_to_test = [
            Models.RAFT_S,
            Models.RAFT_L,
        ]

    def test_raft_custom_in_channels(self):
        """
        Validate that we can create a RAFT model with custom in_channels.
        """
        for model_type in self.models_to_test:
            with self.subTest(model_type=model_type):
                model_name = str(model_type).lower().replace(".", "_")
                model = models.get(model_name, arch_params=dict(in_channels=1), num_classes=1).eval()
                model(torch.rand(1, 2, 1, 640, 640))

    def test_raft_forward(self):
        """
        Validate that we can create a RAFT model with custom in_channels.
        """
        for model_type in self.models_to_test:
            with self.subTest(model_type=model_type):
                model_name = str(model_type).lower().replace(".", "_")
                model = models.get(model_name, num_classes=1).eval()
                model(torch.rand(1, 2, 3, 640, 640))


if __name__ == "__main__":
    unittest.main()
