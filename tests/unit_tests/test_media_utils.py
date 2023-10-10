import unittest

import numpy as np
import torch

from super_gradients.training.utils.media.image import load_images


class TrainingParamsTest(unittest.TestCase):
    def test_load_images(self):

        # list - numpy
        list_images = [np.zeros((3, 100, 100)) for _ in range(15)]
        loaded_images = load_images(list_images)
        self.assertEqual(len(loaded_images), 15)
        for image in loaded_images:
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (100, 100, 3))

        # numpy - batch
        np_images = np.zeros((15, 3, 100, 100))
        loaded_images = load_images(np_images)
        self.assertEqual(len(loaded_images), 15)
        for image in loaded_images:
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (100, 100, 3))

        # list - torcj
        list_images = [torch.zeros((3, 100, 100)) for _ in range(15)]
        loaded_images = load_images(list_images)
        self.assertEqual(len(loaded_images), 15)
        for image in loaded_images:
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (100, 100, 3))

        # torch - batch
        torch_images = torch.zeros((15, 3, 100, 100))
        loaded_images = load_images(torch_images)
        self.assertEqual(len(loaded_images), 15)
        for image in loaded_images:
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (100, 100, 3))


if __name__ == "__main__":
    unittest.main()
