import unittest

import torch

from super_gradients.common.object_names import Models
from super_gradients.training import models


class TestYOLONAS(unittest.TestCase):
    def setUp(self):
        pass

    def test_yolo_nas_custom_in_channels(self):
        """
        Validate that we can create a YOLO-NAS model with custom in_channels.
        """
        model = models.get(Models.YOLO_NAS_S, arch_params=dict(in_channels=2), num_classes=17)
        model(torch.rand(1, 2, 640, 640))


if __name__ == "__main__":
    unittest.main()
