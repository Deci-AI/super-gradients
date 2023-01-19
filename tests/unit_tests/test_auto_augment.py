import unittest
import torchvision.transforms as transforms
from super_gradients.training.datasets.auto_augment import RandAugment
from super_gradients.training.datasets.datasets_utils import get_color_augmentation
import numpy as np
from PIL import Image


class TestAutoAugment(unittest.TestCase):
    def setUp(self):
        self.dataset_params = {"batch_size": 1, "color_jitter": 0.1, "rand_augment_config_string": "m9-mstd0.5"}

    def test_autoaugment_call(self):
        """
        tests a simple call to auto augment and other augmentations and verifies image size
        """
        image_size = 224
        color_augmentation = get_color_augmentation("m9-mstd0.5", color_jitter=None, crop_size=image_size)
        self.assertTrue(isinstance(color_augmentation, RandAugment))
        img = Image.fromarray(np.ones((image_size, image_size, 3)).astype("uint8"))
        augmented_image = color_augmentation(img)
        self.assertTrue(augmented_image.size == (image_size, image_size))

        color_augmentation = get_color_augmentation(None, color_jitter=(0.7, 0.7, 0.7), crop_size=image_size)
        self.assertTrue(isinstance(color_augmentation, transforms.ColorJitter))
        img = Image.fromarray(np.random.randn(image_size, image_size, 3).astype("uint8"))
        augmented_image = color_augmentation(img)
        self.assertTrue(augmented_image.size == (image_size, image_size))


if __name__ == "__main__":
    unittest.main()
