import unittest
from pathlib import Path

from super_gradients.training.datasets import Cifar10, Cifar100, ImageNetDataset, COCODetectionDataset, CoCoSegmentationDataSet
from albumentations import Compose, HorizontalFlip, InvertImg
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_image(image):
    """
    Visualize the input image.

    :param image: torch.Tensor representing the input image with values between 0 and 1. Shape: (C, H, W).
    """
    # Convert torch tensor to numpy array
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Change shape from (C, H, W) to (H, W, C)

    # Display the image
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()


def visualize_mask(mask, num_classes=None, class_colors=None):
    """
    Visualize the segmentation mask.

    :param mask: torch.Tensor representing the segmentation mask with class indices. Shape: (H, W).
    :param num_classes: Number of classes in the segmentation mask.
    :param class_colors: A dictionary mapping class indices to RGB color values.
    """
    # Convert torch tensor to numpy array
    mask_np = mask.cpu().numpy()

    # Determine the number of classes
    if num_classes is None:
        num_classes = int(torch.max(mask) + 1)

    # Define default class colors if not provided
    if class_colors is None:
        class_colors = {i: plt.cm.tab10(i)[:-1] for i in range(num_classes)}  # Exclude the alpha channel

    # Create a colormap for visualization
    colormap = ListedColormap([class_colors[i] for i in range(num_classes)])

    # Display the mask
    plt.imshow(mask_np, cmap=colormap)
    plt.colorbar(ticks=range(num_classes))
    plt.axis("off")
    plt.show()


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

    def test_coco_segmentation_albumentations_intergration(self):
        mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "tinycoco")
        ds = CoCoSegmentationDataSet(
            root_dir=mini_coco_data_dir,
            list_file="instances_val2017.json",
            samples_sub_directory="images/val2017",
            targets_sub_directory="annotations",
            transforms=[
                {"SegRescale": {"short_size": 512}},
                {
                    "SegCropImageAndMask": {"crop_size": 256, "mode": "center"},
                },
                {
                    "Albumentations": {
                        "Compose": {"transforms": [{"HorizontalFlip": {"p": 0.5}}, {"RandomBrightnessContrast": {"p": 0.5}}]},
                    }
                },
                "SegToTensor",
            ],
        )
        image, mask = ds[3]
        visualize_image(image)
        visualize_mask(mask, num_classes=len(ds.classes))


if __name__ == "__main__":
    unittest.main()
