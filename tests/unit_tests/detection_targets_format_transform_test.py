import numpy as np
import unittest

from super_gradients.training.transforms.transforms import DetectionTargetsFormatTransform

from super_gradients.training.datasets.data_formats.default_formats import (
    XYXY_LABEL,
    LABEL_XYXY,
    LABEL_CXCYWH,
    LABEL_NORMALIZED_XYXY,
    LABEL_NORMALIZED_CXCYWH,
)


class DetectionTargetsTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((3, 100, 200))

    def test_label_first_2_label_last(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        output = np.array([[50, 10, 20, 30, 40]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=XYXY_LABEL, output_format=LABEL_XYXY)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_xyxy_2_normalized_xyxy(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        _, h, w = self.image.shape
        output = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_XYXY, output_format=LABEL_NORMALIZED_XYXY)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_xyxy_2_cxcywh(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        _, h, w = self.image.shape
        output = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_XYXY, output_format=LABEL_CXCYWH)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_xyxy_2_normalized_cxcywh(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        _, h, w = self.image.shape
        output = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_XYXY, output_format=LABEL_NORMALIZED_CXCYWH)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_normalized_xyxy_2_cxcywh(self):
        _, h, w = self.image.shape
        input = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        output = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_NORMALIZED_XYXY, output_format=LABEL_CXCYWH)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_normalized_xyxy_2_normalized_cxcywh(self):
        _, h, w = self.image.shape
        input = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        output = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_NORMALIZED_XYXY, output_format=LABEL_NORMALIZED_CXCYWH)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_cxcywh_2_xyxy(self):
        output = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        input = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_CXCYWH, output_format=LABEL_XYXY)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_cxcywh_2_normalized_xyxy(self):
        _, h, w = self.image.shape
        output = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        input = np.array([[10, 30, 40, 20, 20]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_CXCYWH, output_format=LABEL_NORMALIZED_XYXY)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_normalized_cxcywh_2_xyxy(self):
        _, h, w = self.image.shape
        input = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        output = np.array([[10, 20, 30, 40, 50]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_NORMALIZED_CXCYWH, output_format=LABEL_XYXY)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))

    def test_normalized_cxcywh_2_normalized_xyxy(self):
        _, h, w = self.image.shape
        output = np.array([[10, 20 / w, 30 / h, 40 / w, 50 / h]], dtype=np.float32)
        input = np.array([[10, 30 / w, 40 / h, 20 / w, 20 / h]], dtype=np.float32)
        sample = {"image": self.image, "target": input}

        transform = DetectionTargetsFormatTransform(input_dim=self.image.shape[1:], input_format=LABEL_NORMALIZED_CXCYWH, output_format=LABEL_NORMALIZED_XYXY)
        t_output = transform(sample)["target"]
        self.assertTrue(np.allclose(output, t_output, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
