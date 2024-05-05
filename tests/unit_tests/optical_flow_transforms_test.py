import unittest

import numpy as np

from super_gradients.training.samples import OpticalFlowSample
from super_gradients.training.transforms.transforms import (
    OpticalFlowColorJitter,
    OpticalFlowOcclusion,
    OpticalFlowRandomRescale,
    OpticalFlowRandomFlip,
    OpticalFlowCrop,
    OpticalFlowNormalize,
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

        expected_shape_images = (2, int(round(self.sample.images.shape[1] * scale_factor)), int(round(self.sample.images.shape[2] * scale_factor)), 3)
        expected_shape_flow = (int(round(self.sample.flow_map.shape[0] * scale_factor)), int(round(self.sample.flow_map.shape[1] * scale_factor)), 2)
        expected_shape_valid = (int(round(self.sample.valid.shape[0] * scale_factor)), int(round(self.sample.valid.shape[1] * scale_factor)))

        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, expected_shape_images)

        self.assertIsInstance(transformed_sample.flow_map, np.ndarray)
        self.assertEqual(transformed_sample.flow_map.shape, expected_shape_flow)

        self.assertIsInstance(transformed_sample.valid, np.ndarray)
        self.assertEqual(transformed_sample.valid.shape, expected_shape_valid)

    def test_OpticalFlowRandomFlip(self):
        transform = OpticalFlowRandomFlip(h_flip_prob=0.5, v_flip_prob=0.1)
        transformed_sample = transform(self.sample)

        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, self.sample.images.shape)

        self.assertIsInstance(transformed_sample.flow_map, np.ndarray)
        self.assertEqual(transformed_sample.flow_map.shape, self.sample.flow_map.shape)

        self.assertIsInstance(transformed_sample.valid, np.ndarray)
        self.assertEqual(transformed_sample.valid.shape, self.sample.valid.shape)

    def test_OpticalFlowCrop(self):
        transform = OpticalFlowCrop(crop_size=(50, 50), mode="random")
        transformed_sample = transform(self.sample)
        self.assertIsInstance(transformed_sample.images, np.ndarray)
        self.assertEqual(transformed_sample.images.shape, (2, 50, 50, 3))

    def test_OpticalFlowNormalize(self):
        transform = OpticalFlowNormalize()
        transformed_sample = transform(self.sample)

        # Check if normalization is applied correctly
        self.assertTrue(np.allclose(transformed_sample.images, self.sample.images / 255.0))
        self.assertTrue(np.allclose(transformed_sample.flow_map, self.sample.flow_map))
        self.assertTrue(np.array_equal(transformed_sample.valid, self.sample.valid))


if __name__ == "__main__":
    unittest.main()
