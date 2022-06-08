import numpy as np
import unittest

from super_gradients.training.transforms.transforms import DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionTargetsFormat

class DetectionTargetsTransformTest(unittest.TestCase):
    def test_label_first_2_label_last(self):
        image = np.zeros((3,100,200))
        input = np.array([[10, 20, 30, 40, 50]])
        output = np.array([[50, 10, 20,30, 40]])
        transform = DetectionTargetsFormatTransform(max_targets=1,
                                                    input_format=DetectionTargetsFormat.XYXY_LABEL,
                                                    output_format=DetectionTargetsFormat.LABEL_XYXY)
        sample = {"image": image, "target": input}
        self.assertTrue(np.array_equal(transform(sample)["target"], output))






if __name__ == '__main__':
    unittest.main()
