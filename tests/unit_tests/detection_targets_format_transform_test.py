import numpy as np
import unittest

from super_gradients.training.transforms.transforms import DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat


class DetectionTargetsTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((3, 100, 200))

    def test_label_first_2_label_last(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        output = np.array([[50, 10, 20, 30, 40]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.XYXY_LABEL,
                                                    output_format=DetectionTargetsFormat.LABEL_XYXY)
        sample = {"image": self.image, "target": input}
        self.assertTrue(np.array_equal(transform(sample)["target"], output))

    def test_xyxy_2_normalized_xyxy(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        _, h, w = self.image.shape
        output = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_XYXY,
                                                    output_format=DetectionTargetsFormat.LABEL_NORMALIZED_XYXY)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.array_equal(output, t_output))

    def test_xyxy_2_cxcywh(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        _, h, w = self.image.shape
        output = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_XYXY,
                                                    output_format=DetectionTargetsFormat.LABEL_CXCYWH)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.array_equal(output, t_output))

    def test_xyxy_2_normalized_cxcywh(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        _, h, w = self.image.shape
        output = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_XYXY,
                                                    output_format=DetectionTargetsFormat.LABEL_NORMALIZED_CXCYWH)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.array_equal(output, t_output))

    def test_normalized_xyxy_2_cxcywh(self):
        _, h, w = self.image.shape
        input = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        output = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_NORMALIZED_XYXY,
                                                    output_format=DetectionTargetsFormat.LABEL_CXCYWH)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output))

    def test_normalized_xyxy_2_normalized_cxcywh(self):
        _, h, w = self.image.shape
        input = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        output = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_NORMALIZED_XYXY,
                                                    output_format=DetectionTargetsFormat.LABEL_NORMALIZED_CXCYWH)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output))

    def test_cxcywh_2_xyxy(self):
        output = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        input = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_CXCYWH,
                                                    output_format=DetectionTargetsFormat.LABEL_XYXY)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.array_equal(output, t_output))

    def test_cxcywh_2_normalized_xyxy(self):
        _, h, w = self.image.shape
        output = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        input = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_CXCYWH,
                                                    output_format=DetectionTargetsFormat.LABEL_NORMALIZED_XYXY)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.array_equal(output, t_output))

    def test_normalized_cxcywh_2_xyxy(self):
        _, h, w = self.image.shape
        input = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        output = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_NORMALIZED_CXCYWH,
                                                    output_format=DetectionTargetsFormat.LABEL_XYXY)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output))

    def test_normalized_cxcywh_2_normalized_xyxy(self):
        _, h, w = self.image.shape
        output = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        input = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_NORMALIZED_CXCYWH,
                                                    output_format=DetectionTargetsFormat.LABEL_NORMALIZED_XYXY)
        sample = {"image": self.image, "target": input}
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output))


if __name__ == '__main__':
    unittest.main()
