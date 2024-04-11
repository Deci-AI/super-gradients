import unittest

import numpy as np
import torch

from super_gradients.training.samples import OpticalFlowSample
from super_gradients.training.transforms.transforms import (
    OpticalFlowColorJitter,
    OpticalFlowOcclusion,
    OpticalFlowRandomRescale,
    OpticalFlowRandomFlip,
    OpticalFlowCrop,
    OpticalFlowToTensor,
)


class OpticalFlowTransformsTest(unittest.TestCase):
    def setUp(self):
        # Create an OpticalFlowSample
        h, w = 400, 400
        img1 = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        flow_map = np.random.randn(h, w, 2)
        valid = (np.abs(flow_map[:, :, 0]) < 1000) & (np.abs(flow_map[:, :, 1]) < 1000)

        self.sample = OpticalFlowSample(images=np.stack([img1, img2]), flow_map=flow_map, valid=valid)

    def test_OpticalFlowColorJitter(self):
        transform = OpticalFlowColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.16, prob=1.0)
        transformed_sample = transform(self.sample)
        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, self.sample.images.shape)

    def test_OpticalFlowOcclusion(self):
        transform = OpticalFlowOcclusion(prob=1.0, bounds=(10, 30))
        transformed_sample = transform(self.sample)
        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, self.sample.images.shape)

    def test_OpticalFlowRandomRescale(self):
        transform = OpticalFlowRandomRescale(min_scale=0.9, max_scale=1.2, prob=1.0)
        transformed_sample = transform(self.sample)

        scale_factor = transform.scale

        expected_shape = (2, int(round(self.sample.images.shape[1] * scale_factor)), int(round(self.sample.images.shape[2] * scale_factor)), 3)

        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, expected_shape)

    def test_OpticalFlowRandomFlip(self):
        transform = OpticalFlowRandomFlip(h_flip_prob=0.5, v_flip_prob=0.1)
        transformed_sample = transform(self.sample)
        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, self.sample.images.shape)

    def test_OpticalFlowCrop(self):
        transform = OpticalFlowCrop(crop_size=(50, 50), mode="random")
        transformed_sample = transform(self.sample)
        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, (2, 50, 50, 3))

    def test_OpticalFlowToTensor(self):
        transform = OpticalFlowToTensor()
        transformed_sample = transform(self.sample)
        self.assertIsInstance(transformed_sample.images, torch.Tensor)
        self.assertIsInstance(transformed_sample.flow_map, torch.Tensor)
        self.assertIsInstance(transformed_sample.valid, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
