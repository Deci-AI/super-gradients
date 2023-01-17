import unittest

import torch

from super_gradients.modules import RepVGGBlock


class TestRepVGGBlock(unittest.TestCase):
    def test_dilation_padding(self):
        in_channels = 16
        out_channels = 27
        feature_map = torch.randn((4, in_channels, 256, 256))
        block = RepVGGBlock(in_channels=feature_map.size(1), out_channels=out_channels, dilation=4)

        outputs = block(feature_map)

        self.assertEqual(outputs.size(1), out_channels)
        self.assertEqual(outputs.size(2), feature_map.size(2))
        self.assertEqual(outputs.size(3), feature_map.size(3))


if __name__ == "__main__":
    unittest.main()
