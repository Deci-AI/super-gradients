import unittest

import torch

from super_gradients.training.models.detection_models.yolox import YoloX_N, YoloX_T, YoloX_S, YoloX_M, YoloX_L, YoloX_X
from super_gradients.training.utils.utils import HpmStruct


class TestYOLOX(unittest.TestCase):
    def setUp(self) -> None:
        self.arch_params = HpmStruct(num_classes=10)
        self.yolo_classes = [YoloX_N, YoloX_T, YoloX_S, YoloX_M, YoloX_L, YoloX_X]

    def test_yolox_creation(self):
        """
        test_yolox_creation - Tests the creation of the models
            :return:
        """
        dummy_input = torch.randn(1, 3, 320, 320)

        with torch.no_grad():

            for yolo_cls in self.yolo_classes:
                yolo_model = yolo_cls(self.arch_params)
                # THIS SHOULD RUN THE FORWARD ONCE
                yolo_model.eval()
                output_standard = yolo_model(dummy_input)
                self.assertIsNotNone(output_standard)

                # THIS SHOULD RUN A TRAINING FORWARD
                yolo_model.train()
                output_train = yolo_model(dummy_input)

                self.assertIsNotNone(output_train)

                # THIS SHOULD RUN THE FORWARD AUGMENT
                yolo_model.eval()
                yolo_model.augmented_inference = True
                output_augment = yolo_model(dummy_input)
                self.assertIsNotNone(output_augment)


if __name__ == "__main__":
    unittest.main()
