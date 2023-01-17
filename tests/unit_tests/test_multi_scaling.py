import unittest

import torch
from super_gradients.training.datasets.datasets_utils import DetectionMultiscalePrePredictionCallback


class MultiScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.size = (1024, 512)
        self.batch_size = 12
        self.change_frequency = 10
        self.multiscale_callback = DetectionMultiscalePrePredictionCallback(change_frequency=self.change_frequency)

    def _create_batch(self):
        inputs = torch.rand((self.batch_size, 3, self.size[0], self.size[1])) * 255
        targets = torch.cat([torch.tensor([[[0, 0, 10, 10, 0]]]) for _ in range(self.batch_size)], 0)
        return inputs, targets

    def test_multiscale_keep_state(self):
        """Check that the multiscale keeps in memory the new size to use between the size swaps"""

        for i in range(5):
            post_multiscale_input_shapes = []
            for j in range(self.change_frequency):
                inputs, targets = self._create_batch()
                post_multiscale_input, _ = self.multiscale_callback(inputs, targets, batch_idx=i * self.change_frequency + j)
                post_multiscale_input_shapes.append(list(post_multiscale_input.shape))

                # The shape should be the same for a given between k * self.change_frequency and (k+1)*self.change_frequency
                self.assertListEqual(post_multiscale_input_shapes[0], post_multiscale_input_shapes[-1])


if __name__ == "__main__":
    unittest.main()
