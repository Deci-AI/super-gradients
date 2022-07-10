import unittest

import numpy as np
import torch
import torch.nn as nn

from super_gradients.training.models.detection_models.yolov5 import YoLoV5N, YoLoV5S, YoLoV5M, YoLoV5L, YoLoV5X, Custom_YoLoV5
from super_gradients.training.utils.utils import HpmStruct


class TestYoloV5(unittest.TestCase):
    def setUp(self) -> None:
        self.arch_params = HpmStruct(num_classes=10)
        self.yolo_classes = [YoLoV5N, YoLoV5S, YoLoV5M, YoLoV5L, YoLoV5X, Custom_YoLoV5]

    def test_yolov5_creation(self):
        """
        test_yolov5_creation - Tests the creation of the model class itself
            :return:
        """
        dummy_input = torch.randn(1, 3, 320, 320)

        with torch.no_grad():
            # DEFAULT Custom_YoLoV5
            yolov5_custom = Custom_YoLoV5(arch_params=self.arch_params)
            self.assertTrue(isinstance(yolov5_custom._nms, nn.modules.linear.Identity))
            self.assertIsNotNone(yolov5_custom(dummy_input))

            # add NMS to the model
            yolov5_custom = Custom_YoLoV5(arch_params=HpmStruct(num_classes=10, add_nms=True))
            self.assertFalse(isinstance(yolov5_custom._nms, nn.modules.linear.Identity))
            self.assertIsNotNone(yolov5_custom(dummy_input))

            # fuse_conv_and_bn layers
            yolov5_custom = Custom_YoLoV5(arch_params=HpmStruct(num_classes=10, add_nms=False, fuse_conv_and_bn=True))
            self.assertIsNotNone(yolov5_custom(dummy_input))

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

    def test_init_param_groups(self):
        train_params = HpmStruct(optimizer_params={'weight_decay': 0.01})
        for yolo_cls in self.yolo_classes:
            yolo_model = yolo_cls(self.arch_params)
            yolo_model.train()

            params_total = sum(p.numel() for p in yolo_model.parameters() if p.requires_grad)
            param_groups = yolo_model.initialize_param_groups(0.1, train_params)
            optimizer_params_total = sum(p.numel() for g in param_groups for _, p in g['named_params'])

            self.assertEqual(params_total, optimizer_params_total)

    def test_custom_width_mult(self):
        """
        Test that yolo can be created with various width multiplies without rounding issues
        """
        dummy_input = torch.randn(1, 3, 320, 320)

        with torch.no_grad():
            for width in np.arange(0.77, 1.4, 0.07):
                yolov5_custom = Custom_YoLoV5(arch_params=HpmStruct(num_classes=10, width_mult_factor=width))
                self.assertIsNotNone(yolov5_custom(dummy_input))


if __name__ == '__main__':
    unittest.main()
