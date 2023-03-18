import unittest

import numpy as np

from super_gradients.training.transforms.keypoint_transforms import (
    KeypointsRandomHorizontalFlip,
    KeypointsRandomVerticalFlip,
    KeypointsRandomAffineTransform,
    KeypointsPadIfNeeded,
    KeypointsLongestMaxSize,
)
from super_gradients.training.transforms.transforms import DetectionImagePermute, DetectionPadToSize


class TestTransforms(unittest.TestCase):
    def test_keypoints_random_affine(self):
        image = np.random.rand(640, 480, 3)
        mask = np.random.rand(640, 480)

        # Cover all image pixels with keypoints. This would guarantee test coverate of all possible keypoint locations
        # without relying on randomly generated keypoints.
        x = np.arange(image.shape[1])
        y = np.arange(image.shape[0])
        xv, yv = np.meshgrid(x, y, indexing="xy")

        joints = np.stack([xv.flatten(), yv.flatten(), np.ones_like(yv.flatten())], axis=-1)  # [N, 3]
        joints = joints.reshape((-1, 1, 3)).repeat(17, axis=1)  # [N, 17, 3]

        aug = KeypointsRandomAffineTransform(min_scale=0.8, max_scale=1.2, max_rotation=30, max_translate=0.5, prob=1, image_pad_value=0, mask_pad_value=0)
        aug_image, aug_mask, aug_joints, _, _ = aug(image, mask, joints, None, None)

        joints_outside_image = (
            (aug_joints[:, :, 0] < 0) | (aug_joints[:, :, 1] < 0) | (aug_joints[:, :, 0] >= aug_image.shape[1]) | (aug_joints[:, :, 1] >= aug_image.shape[0])
        )

        # Ensure that keypoints outside the image are not visible
        self.assertTrue((aug_joints[joints_outside_image, 2] == 0).all(), msg=f"{aug_joints[joints_outside_image]}")
        # Ensure that all keypoints with visible status are inside the image
        # (There is no intersection of two sets: keypoints outside the image and keypoints with visible status)
        self.assertFalse((joints_outside_image & (aug_joints[:, :, 2] == 1)).any())

    def test_keypoints_horizontal_flip(self):
        image = np.random.rand(640, 480, 3)
        mask = np.random.rand(640, 480)
        joints = np.random.randint(0, 100, size=(1, 17, 3))

        aug = KeypointsRandomHorizontalFlip(flip_index=[16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], prob=1)
        aug_image, aug_mask, aug_joints, _, _ = aug(image, mask, joints, None, None)

        np.testing.assert_array_equal(aug_image, image[:, ::-1, :])
        np.testing.assert_array_equal(aug_mask, mask[:, ::-1])
        np.testing.assert_array_equal(image.shape[1] - aug_joints[:, ::-1, 0] - 1, joints[..., 0])
        np.testing.assert_array_equal(aug_joints[:, ::-1, 1], joints[..., 1])
        np.testing.assert_array_equal(aug_joints[:, ::-1, 2], joints[..., 2])

    def test_keypoints_vertical_flip(self):
        image = np.random.rand(640, 480, 3)
        mask = np.random.rand(640, 480)
        joints = np.random.randint(0, 100, size=(1, 17, 3))

        aug = KeypointsRandomVerticalFlip(prob=1)
        aug_image, aug_mask, aug_joints, _, _ = aug(image, mask, joints, None, None)

        np.testing.assert_array_equal(aug_image, image[::-1, :, :])
        np.testing.assert_array_equal(aug_mask, mask[::-1, :])
        np.testing.assert_array_equal(aug_joints[..., 0], joints[..., 0])
        np.testing.assert_array_equal(image.shape[0] - aug_joints[..., 1] - 1, joints[..., 1])
        np.testing.assert_array_equal(aug_joints[..., 2], joints[..., 2])

    def test_keypoints_pad_if_needed(self):
        image = np.random.rand(640, 480, 3)
        mask = np.random.rand(640, 480)
        joints = np.random.randint(0, 100, size=(1, 17, 3))

        aug = KeypointsPadIfNeeded(min_width=768, min_height=768, image_pad_value=0, mask_pad_value=0)
        aug_image, aug_mask, aug_joints, _, _ = aug(image, mask, joints, None, None)

        self.assertEqual(aug_image.shape, (768, 768, 3))
        self.assertEqual(aug_mask.shape, (768, 768))
        np.testing.assert_array_equal(aug_joints, joints)

    def test_keypoints_longest_max_size(self):
        image = np.random.rand(640, 480, 3)
        mask = np.random.rand(640, 480)
        joints = np.random.randint(0, 480, size=(1, 17, 3))

        aug = KeypointsLongestMaxSize(max_height=512, max_width=512)
        aug_image, aug_mask, aug_joints, _, _ = aug(image, mask, joints, None, None)

        self.assertEqual(aug_image.shape[:2], aug_mask.shape[:2])
        self.assertLessEqual(aug_image.shape[0], 512)
        self.assertLessEqual(aug_image.shape[1], 512)

        self.assertTrue((aug_joints[..., 0] < aug_image.shape[1]).all())
        self.assertTrue((aug_joints[..., 1] < aug_image.shape[0]).all())

    def test_detection_image_permute(self):
        aug = DetectionImagePermute(dims=(2, 1, 0))
        image = np.random.rand(640, 480, 3)
        sample = {"image": image}

        output = aug(sample)
        self.assertEqual(output["image"].shape, (3, 480, 640))

    def test_detection_pad_to_size(self):
        aug = DetectionPadToSize(output_size=(640, 640), pad_value=123)
        image = np.ones((512, 480, 3))

        # Boxes in format (x1, y1, x2, y2, class_id)
        boxes = np.array([[0, 0, 100, 100, 0], [100, 100, 200, 200, 1]])

        sample = {"image": image, "target": boxes}
        output = aug(sample)

        shift_x = (640 - 480) // 2
        shift_y = (640 - 512) // 2
        expected_boxes = np.array(
            [[0 + shift_x, 0 + shift_y, 100 + shift_x, 100 + shift_y, 0], [100 + shift_x, 100 + shift_y, 200 + shift_x, 200 + shift_y, 1]]
        )
        self.assertEqual(output["image"].shape, (640, 640, 3))
        np.testing.assert_array_equal(output["target"], expected_boxes)


if __name__ == "__main__":
    unittest.main()
