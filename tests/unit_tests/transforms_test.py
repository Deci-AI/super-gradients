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

from super_gradients.training.transforms.utils import (
    _rescale_image,
    _rescale_bboxes,
    _pad_image,
    _shift_bboxes,
    _rescale_and_pad_to_size,
    _rescale_xyxy_bboxes,
    _get_center_padding_coordinates,
    _get_bottom_right_padding_coordinates,
    PaddingCoordinates,
)


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

    def test_rescale_image(self):
        image = np.random.randint(0, 256, size=(640, 480, 3), dtype=np.uint8)
        target_shape = (320, 240)
        rescaled_image = _rescale_image(image, target_shape)

        # Check if the rescaled image has the correct target shape
        self.assertEqual(rescaled_image.shape[:2], target_shape)

    def test_rescale_bboxes(self):
        sy, sx = (2.0, 0.5)

        # Empty bboxes
        bboxes = np.zeros((0, 4))
        expected_bboxes = np.zeros((0, 4))
        rescaled_bboxes = _rescale_bboxes(targets=bboxes, scale_factors=(sy, sx))
        np.testing.assert_array_equal(rescaled_bboxes, expected_bboxes)

        # Not empty bboxes
        bboxes = np.array([[10, 20, 50, 60, 1], [30, 40, 80, 90, 2]], dtype=np.float32)
        expected_bboxes = np.array([[5.0, 40.0, 25.0, 120.0, 1.0], [15.0, 80.0, 40.0, 180.0, 2.0]], dtype=np.float32)
        rescaled_bboxes = _rescale_bboxes(targets=bboxes, scale_factors=(sy, sx))
        np.testing.assert_array_equal(rescaled_bboxes, expected_bboxes)

    def test_pad_image(self):
        image = np.random.randint(0, 256, size=(640, 480, 3), dtype=np.uint8)
        padding_coordinates = PaddingCoordinates(top=80, bottom=80, left=60, right=60)
        pad_value = 0
        shifted_image = _pad_image(image, padding_coordinates, pad_value)

        # Check if the shifted image has the correct shape
        self.assertEqual(shifted_image.shape, (800, 600, 3))
        # Check if the padding values are correct
        self.assertTrue((shifted_image[: padding_coordinates.top, :, :] == pad_value).all())
        self.assertTrue((shifted_image[-padding_coordinates.bottom :, :, :] == pad_value).all())
        self.assertTrue((shifted_image[:, : padding_coordinates.left, :] == pad_value).all())
        self.assertTrue((shifted_image[:, -padding_coordinates.right :, :] == pad_value).all())

    def test_shift_bboxes(self):
        bboxes = np.array([[10, 20, 50, 60, 1], [30, 40, 80, 90, 2]], dtype=np.float32)
        shift_w, shift_h = 60, 80
        shifted_bboxes = _shift_bboxes(bboxes, shift_w, shift_h)

        # Check if the shifted bboxes have the correct values
        expected_bboxes = np.array([[70, 100, 110, 140, 1], [90, 120, 140, 170, 2]], dtype=np.float32)
        np.testing.assert_array_equal(shifted_bboxes, expected_bboxes)

    def test_rescale_xyxy_bboxes(self):
        bboxes = np.array([[10, 20, 50, 60, 1], [30, 40, 80, 90, 2]], dtype=np.float32)
        r = 0.5
        rescaled_bboxes = _rescale_xyxy_bboxes(bboxes, r)

        # Check if the rescaled bboxes have the correct values
        expected_bboxes = np.array([[5.0, 10.0, 25.0, 30.0, 1.0], [15.0, 20.0, 40.0, 45.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(rescaled_bboxes, expected_bboxes)

    def test_padding(self):
        # Test Case 1: Padding needed
        image = np.array([[1, 2], [3, 4]])
        padding_coordinates = PaddingCoordinates(top=0, left=0, bottom=1, right=2)
        expected_padded_image = np.array([[1, 2, 114, 114], [3, 4, 114, 114], [114, 114, 114, 114]])

        padded_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=114)
        np.testing.assert_array_equal(padded_image, expected_padded_image)

        # Test Case 2: No padding needed
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        padding_coordinates = PaddingCoordinates(top=0, left=0, bottom=0, right=0)
        expected_padded_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        padded_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=114)
        np.testing.assert_array_equal(padded_image, expected_padded_image)

        # Test Case 3: Image with channel dimension
        image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        padding_coordinates = PaddingCoordinates(top=0, left=0, bottom=1, right=2)
        expected_padded_image = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]],
                [[7, 8, 9], [10, 11, 12], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        )

        padded_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=0)
        np.testing.assert_array_equal(padded_image, expected_padded_image)

    def test_get_padding_coordinates(self):
        # Test Case 1: Width padding required
        image = np.zeros((640, 480))
        output_size = (640, 640)
        expected_center_padding = PaddingCoordinates(top=0, bottom=0, left=80, right=80)
        expected_bottom_right_padding = PaddingCoordinates(top=0, bottom=0, left=0, right=160)

        center_padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        bottom_right_padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        self.assertEqual(center_padding_coordinates, expected_center_padding)
        self.assertEqual(bottom_right_padding_coordinates, expected_bottom_right_padding)

        # Test Case 2: Height padding required
        image = np.zeros((480, 640))
        output_size = (640, 640)
        expected_center_padding = PaddingCoordinates(top=80, bottom=80, left=0, right=0)
        expected_bottom_right_padding = PaddingCoordinates(top=0, bottom=160, left=0, right=0)

        center_padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        bottom_right_padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        self.assertEqual(center_padding_coordinates, expected_center_padding)
        self.assertEqual(bottom_right_padding_coordinates, expected_bottom_right_padding)

        # Test Case 3: Width and Height padding required
        image = np.zeros((480, 640))
        output_size = (800, 800)
        expected_center_padding = PaddingCoordinates(top=160, bottom=160, left=80, right=80)
        expected_bottom_right_padding = PaddingCoordinates(top=0, bottom=320, left=0, right=160)

        center_padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        bottom_right_padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        self.assertEqual(center_padding_coordinates, expected_center_padding)
        self.assertEqual(bottom_right_padding_coordinates, expected_bottom_right_padding)

        # Test Case 4: Image shape is bigger than output shape
        image = np.zeros((800, 800))
        output_size = (640, 640)
        expected_center_padding = PaddingCoordinates(top=-80, bottom=-80, left=-80, right=-80)
        expected_bottom_right_padding = PaddingCoordinates(top=0, bottom=-160, left=0, right=-160)

        center_padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        bottom_right_padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        self.assertEqual(center_padding_coordinates, expected_center_padding)
        self.assertEqual(bottom_right_padding_coordinates, expected_bottom_right_padding)

        # Test Case 5: Width and Height padding required with an image of 3 channels
        image = np.zeros((480, 640, 3))
        output_size = (800, 800)
        expected_center_padding = PaddingCoordinates(top=160, bottom=160, left=80, right=80)
        expected_bottom_right_padding = PaddingCoordinates(top=0, bottom=320, left=0, right=160)

        center_padding_coordinates = _get_center_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        bottom_right_padding_coordinates = _get_bottom_right_padding_coordinates(input_shape=image.shape, output_shape=output_size)
        self.assertEqual(center_padding_coordinates, expected_center_padding)
        self.assertEqual(bottom_right_padding_coordinates, expected_bottom_right_padding)

    def test_rescale_and_pad_to_size(self):
        image = np.random.randint(0, 256, size=(640, 480, 3), dtype=np.uint8)
        output_size = (800, 500)
        pad_val = 114
        rescaled_padded_image, r = _rescale_and_pad_to_size(image, output_size, pad_val=pad_val)

        # Check if the rescaled and padded image has the correct shape
        self.assertEqual(rescaled_padded_image.shape, (3, *output_size))

        # Check if the image is rescaled with the correct ratio
        resized_image_shape = (int(image.shape[0] * r), int(image.shape[1] * r))

        # Check if the padding is correctly applied
        padded_area = rescaled_padded_image[:, resized_image_shape[0] :, :]  # Right padding area
        self.assertTrue((padded_area == pad_val).all())
        padded_area = rescaled_padded_image[:, :, resized_image_shape[1] :]  # Bottom padding area
        self.assertTrue((padded_area == pad_val).all())


if __name__ == "__main__":
    unittest.main()
