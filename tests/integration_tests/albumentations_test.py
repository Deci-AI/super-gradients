import os
import unittest
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from albumentations import Compose, HorizontalFlip, InvertImg

from super_gradients.training.datasets import Cifar10, Cifar100, ImageNetDataset, COCODetectionDataset, CoCoSegmentationDataSet, COCOPoseEstimationDataset
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy
from super_gradients.training.datasets.depth_estimation_datasets import NYUv2DepthEstimationDataset


def visualize_image(image):
    """
    Visualize the input image.

    :param image: torch.Tensor representing the input image with values between 0 and 1. Shape: (C, H, W).
    """
    # Convert torch tensor to numpy array
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # Change shape from (C, H, W) to (H, W, C)

    # Display the image
    plt.imshow(image)
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
                "SegConvertToTensor",
            ],
        )
        image, mask = ds[3]
        visualize_image(image)
        visualize_mask(mask, num_classes=len(ds.classes))

    def test_depth_estimation_albumentations_integration(self):
        mini_nyuv2_data_dir = str(Path(__file__).parent.parent / "data" / "nyu2_mini_test")
        mini_nyuv2_df_path = os.path.join(mini_nyuv2_data_dir, "nyu2_mini_test.csv")

        transforms = [
            {
                "Albumentations": {
                    "Compose": {"transforms": [{"Rotate": {"p": 1.0, "limit": 15}}, {"RandomBrightnessContrast": {"p": 1.0}}]},
                }
            }
        ]

        dataset = NYUv2DepthEstimationDataset(root=mini_nyuv2_data_dir, df_path=mini_nyuv2_df_path, transforms=transforms)
        dataset.plot(max_samples_per_plot=8)

    def test_coco_pose_albumentations_intergration(self):
        mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "pose_minicoco")

        edge_links = [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 6],
            [5, 7],
            [5, 11],
            [6, 8],
            [6, 12],
            [7, 9],
            [8, 10],
            [11, 12],
            [11, 13],
            [12, 14],
            [13, 15],
            [14, 16],
        ]

        edge_colors = [
            [214, 39, 40],  # Nose -> LeftEye
            [148, 103, 189],  # Nose -> RightEye
            [44, 160, 44],  # LeftEye -> RightEye
            [140, 86, 75],  # LeftEye -> LeftEar
            [227, 119, 194],  # RightEye -> RightEar
            [127, 127, 127],  # LeftEar -> LeftShoulder
            [188, 189, 34],  # RightEar -> RightShoulder
            [127, 127, 127],  # Shoulders
            [188, 189, 34],  # LeftShoulder -> LeftElbow
            [140, 86, 75],  # LeftTorso
            [23, 190, 207],  # RightShoulder -> RightElbow
            [227, 119, 194],  # RightTorso
            [31, 119, 180],  # LeftElbow -> LeftArm
            [255, 127, 14],  # RightElbow -> RightArm
            [148, 103, 189],  # Waist
            [255, 127, 14],  # Left Hip -> Left Knee
            [214, 39, 40],  # Right Hip -> Right Knee
            [31, 119, 180],  # Left Knee -> Left Ankle
            [44, 160, 44],  # Right Knee -> Right Ankle
        ]

        keypoint_colors = [
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
            [31, 119, 180],
            [148, 103, 189],
        ]

        from super_gradients.training.transforms import KeypointsRescale, KeypointsPadIfNeeded

        transforms = [
            KeypointsRescale(height=320, width=640),
            {
                "Albumentations": {
                    "Compose": {
                        "transforms": [{"RandomBrightnessContrast": {"p": 1}}, {"RandomCrop": dict(width=300, height=320)}],
                        "bbox_params": {
                            "min_area": 1,
                            "min_visibility": 0,
                            "min_width": 0,
                            "min_height": 0,
                            "check_each_transform": True,
                        },
                        "keypoint_params": {},
                    },
                }
            },
            KeypointsPadIfNeeded(min_height=350, min_width=350, image_pad_value=0, mask_pad_value=0, padding_mode="center"),
        ]

        ds = COCOPoseEstimationDataset(
            data_dir=mini_coco_data_dir,
            images_dir="images/val2017",
            json_file="annotations/person_keypoints_val2017.json",
            include_empty_samples=True,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
            transforms=transforms,
        )

        sample = next(iter(ds))

        bboxes_xyxy = xywh_to_xyxy(bboxes=np.array(sample.bboxes_xywh), image_shape=sample.image.shape)

        image_with_keypoints = PoseVisualization.draw_poses(
            image=np.array(sample.image),
            poses=sample.joints,
            boxes=bboxes_xyxy,
            scores=None,
            is_crowd=sample.is_crowd,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
            show_keypoint_confidence=False,
            joint_thickness=None,
            box_thickness=None,
            keypoint_radius=None,
        )

        visualize_image(image=image_with_keypoints)

        # Make sure we raise an error when using unsupported transforms
        with self.assertRaises(TypeError):
            transforms = [
                KeypointsRescale(height=320, width=640),
                {
                    "Albumentations": {
                        "Compose": {
                            "transforms": [{"HorizontalFlip": {"p": 1}}],
                            "bbox_params": {
                                "min_area": 1,
                                "min_visibility": 0,
                                "min_width": 0,
                                "min_height": 0,
                                "check_each_transform": True,
                            },
                            "keypoint_params": {},
                        },
                    }
                },
                KeypointsPadIfNeeded(min_height=350, min_width=350, image_pad_value=0, mask_pad_value=0, padding_mode="center"),
            ]
            unsupported_ds = COCOPoseEstimationDataset(
                data_dir=mini_coco_data_dir,
                images_dir="images/val2017",
                json_file="annotations/person_keypoints_val2017.json",
                include_empty_samples=True,
                edge_links=edge_links,
                edge_colors=edge_colors,
                keypoint_colors=keypoint_colors,
                transforms=transforms,
            )

            _ = next(iter(unsupported_ds))


if __name__ == "__main__":
    unittest.main()
