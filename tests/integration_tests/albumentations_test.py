import unittest
from pathlib import Path

import numpy as np

from super_gradients.training.datasets import Cifar10, Cifar100, ImageNetDataset, COCODetectionDataset
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

    def test_coco_albumentations_integration(self):
        mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "tinycoco")

        train_dataset_params = {
            "data_dir": mini_coco_data_dir,
            "subdir": "images/train2017",
            "json_file": "instances_train2017.json",
            "cache": False,
            "input_dim": [512, 512],
            "transforms": [
                {"DetectionMosaic": {"input_dim": [640, 640], "prob": 1.0}},
                {
                    "Albumentations": {
                        "Compose": {
                            "transforms": [{"HorizontalFlip": {"p": 0.5}}, {"RandomBrightnessContrast": {"p": 0.5}}],
                            "bbox_params": {"min_area": 1, "min_visibility": 0, "min_width": 0, "min_height": 0, "check_each_transform": True},
                        },
                    }
                },
                {
                    "DetectionMixup": {
                        "input_dim": [640, 640],
                        "mixup_scale": [0.5, 1.5],
                        # random rescale range for the additional sample in mixup
                        "prob": 1.0,  # probability to apply per-sample mixup
                        "flip_prob": 0.5,
                    }
                },
            ],
        }

        ds = COCODetectionDataset(**train_dataset_params)
        ds.plot()


if __name__ == "__main__":
    unittest.main()
