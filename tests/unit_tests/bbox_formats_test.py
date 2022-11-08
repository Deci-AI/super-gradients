import unittest

import torch

from super_gradients.training.utils.bbox_formats import (
    CXCYWHCoordinateFormat,
    NormalizedXYXYCoordinateFormat,
    NormalizedXYWHCoordinateFormat,
    XYWHCoordinateFormat,
    YXYXCoordinateFormat,
    XYXYCoordinateFormat,
    NormalizedCXCYWHCoordinateFormat,
    convert_bboxes,
)


class BBoxFormatsTest(unittest.TestCase):
    def setUp(self):
        # contains all formats
        self.formats = [
            XYWHCoordinateFormat(),
            XYXYCoordinateFormat(),
            YXYXCoordinateFormat(),
            CXCYWHCoordinateFormat(),
            NormalizedXYWHCoordinateFormat(),
            NormalizedXYXYCoordinateFormat(),
            NormalizedCXCYWHCoordinateFormat(),
        ]

    def test_conversion_to_from_is_correct_2d_input(self):
        """
        Check whether bbox format supports 3D input shape as input: [L, 4]
        """
        gt_bboxes = torch.randint(low=0, high=512, size=(8192, 4)).float()
        # Make bboxes in XYXY format and ensure they all of non-zero area with X2>X1 and Y2>Y1
        gt_bboxes[..., 2:4] += gt_bboxes[..., 0:2] + 1

        image_shape = (2048, 1536)

        for fm1 in self.formats:
            input_bboxes = gt_bboxes.clone()
            intermediate_format = fm1.from_xyxy(input_bboxes, image_shape)
            actual_bboxes = fm1.to_xyxy(intermediate_format, image_shape)
            self.assertTrue(torch.allclose(input_bboxes, actual_bboxes, atol=1, rtol=1), msg=f"Format {fm1} failed to pass sanity check")

    def test_conversion_to_from_is_correct_3d_input(self):
        """
        Check whether bbox format supports 3D input shape as input: [B, L, 4]
        """
        gt_bboxes = torch.randint(low=0, high=512, size=(16, 8192, 4)).float()
        # Make bboxes in XYXY format and ensure they all of non-zero area with X2>X1 and Y2>Y1
        gt_bboxes[..., 2:4] += gt_bboxes[..., 0:2] + 1

        image_shape = (2048, 1536)

        for fm1 in self.formats:
            input_bboxes = gt_bboxes.clone()
            intermediate_format = fm1.from_xyxy(input_bboxes, image_shape)
            actual_bboxes = fm1.to_xyxy(intermediate_format, image_shape)
            self.assertTrue(torch.allclose(input_bboxes, actual_bboxes, atol=1, rtol=1), msg=f"Format {fm1} failed to pass sanity check")

    def test_convert_bboxes(self):
        gt_bboxes = torch.randint(low=0, high=512, size=(16, 8192, 4)).float()
        # Make bboxes in XYXY format and ensure they all of non-zero area with X2>X1 and Y2>Y1
        gt_bboxes[..., 2:4] += gt_bboxes[..., 0:2] + 1

        image_shape = (2048, 1536)
        for src_fmt in self.formats:
            for dst_fmt in self.formats:
                input_bboxes = src_fmt.from_xyxy(gt_bboxes, image_shape)
                intermediate_format = convert_bboxes(input_bboxes, image_shape, src_fmt, dst_fmt)
                actual_bboxes = dst_fmt.to_xyxy(intermediate_format, image_shape)
                self.assertTrue(
                    torch.allclose(gt_bboxes, actual_bboxes, atol=1, rtol=1), msg=f"convert_bboxes failed to convert bboxes from {src_fmt} to {dst_fmt}"
                )


if __name__ == "__main__":
    unittest.main()
