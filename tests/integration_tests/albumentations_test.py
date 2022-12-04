import unittest

import numpy as np

from super_gradients.training.datasets import Cifar10, Cifar100, ImageNetDataset
from albumentations import Compose, HorizontalFlip, InvertImg


class AlbumentationsIntegrationTest(unittest.TestCase):
    def _apply_aug(self, img_no_aug):
        pipe = Compose(transforms=[HorizontalFlip(p=1.0), InvertImg(p=1.0)])
        img_no_aug_transformed = pipe(image=np.array(img_no_aug))["image"]
        return img_no_aug_transformed

    def test_cifar10_albumentations_integration(self):
        ds_no_aug = Cifar10(root="./data/cifar10", train=True, download=True)
        img_no_aug, _ = ds_no_aug.__getitem__(0)

        ds = Cifar10(
            root="./data/cifar10",
            train=True,
            download=True,
            transforms={"Albumentations": {"Compose": {"transforms": [{"HorizontalFlip": {"p": 1.0}}, {"InvertImg": {"p": 1.0}}]}}},
        )

        img_aug, _ = ds.__getitem__(0)
        img_no_aug_transformed = self._apply_aug(img_no_aug)

        self.assertTrue(np.allclose(img_no_aug_transformed, img_aug))

    def test_cifar100_albumentations_integration(self):
        ds_no_aug = Cifar100(root="./data/cifar100", train=True, download=True)
        img_no_aug, _ = ds_no_aug.__getitem__(0)

        ds = Cifar100(
            root="./data/cifar100",
            train=True,
            download=True,
            transforms={"Albumentations": {"Compose": {"transforms": [{"HorizontalFlip": {"p": 1}}, {"InvertImg": {"p": 1.0}}]}}},
        )

        img_aug, _ = ds.__getitem__(0)
        img_no_aug_transformed = self._apply_aug(img_no_aug)

        self.assertTrue(np.allclose(img_no_aug_transformed, img_aug))

    def test_imagenet_albumentations_integration(self):
        ds_no_aug = ImageNetDataset(root="/data/Imagenet/val")
        img_no_aug, _ = ds_no_aug.__getitem__(0)

        ds = ImageNetDataset(
            root="/data/Imagenet/val", transforms={"Albumentations": {"Compose": {"transforms": [{"HorizontalFlip": {"p": 1}}, {"InvertImg": {"p": 1.0}}]}}}
        )
        img_aug, _ = ds.__getitem__(0)
        img_no_aug_transformed = self._apply_aug(img_no_aug)

        self.assertTrue(np.allclose(img_no_aug_transformed, img_aug))


if __name__ == "__main__":
    unittest.main()
