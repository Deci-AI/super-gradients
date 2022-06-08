import numpy as np
import unittest

from super_gradients.training.transforms.transforms import DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat

LABEL_XYXY = "LABEL_XYXY"
LABEL_NORMALIZED_XYXY = "LABEL_NORMALIZED_XYXY"
LABEL_CXCYWH = "LABEL_CXCYWH"
LABEL_NORMALIZED_CXCYWH = "LABEL_NORMALIZED_CXCYWH"


class DetectionTargetsTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((3, 100, 200))

    def test_label_first_2_label_last(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=float)
        output = np.array([[50, 10, 20, 30, 40]], dtype=float)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.XYXY_LABEL,
                                                    output_format=DetectionTargetsFormat.LABEL_XYXY)
        sample = {"image": self.image, "target": input}
        self.assertTrue(np.array_equal(transform(sample)["target"], output))

    def test_xyxy_2_normalized_xyxy(self):
        input = np.array([[10, 20, 30, 40, 50]], dtype=float)
        _, h, w = self.image.shape
        output = np.array([[10, 20/w, 30/h, 40/w, 50/h]], dtype=float)
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.LABEL_XYXY,
                                                    output_format=DetectionTargetsFormat.LABEL_NORMALIZED_XYXY)
        sample = {"image": self.image, "target": input}
        print(transform(sample)["target"])
        self.assertTrue(np.array_equal(transform(sample)["target"], output))




if __name__ == '__main__':
    unittest.main()
