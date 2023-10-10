import os
import tempfile
import unittest

import torch

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.losses import YoloXDetectionLoss, YoloXFastDetectionLoss
from super_gradients.training.models.detection_models.yolox import YoloX_N, YoloX_T, YoloX_S, YoloX_M, YoloX_L, YoloX_X
from super_gradients.training.utils.collate_fn import DetectionCollateFN
from super_gradients.training.utils.utils import HpmStruct


class TestYOLOX(unittest.TestCase):
    def setUp(self) -> None:
        self.arch_params = HpmStruct(num_classes=10)
        self.yolo_classes = [YoloX_N, YoloX_T, YoloX_S, YoloX_M, YoloX_L, YoloX_X]
        self.devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    def test_yolox_creation(self):
        """
        test_yolox_creation - Tests the creation of the models
            :return:
        """
        for device in self.devices:
            dummy_input = torch.randn(1, 3, 320, 320).to(device)
            with torch.no_grad():
                for yolo_cls in self.yolo_classes:
                    yolo_model = yolo_cls(self.arch_params).to(device)
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

    def test_yolox_loss(self):
        samples = [
            (torch.zeros((3, 256, 256)), torch.zeros((100, 5))),
            (torch.zeros((3, 256, 256)), torch.zeros((100, 5))),
            (torch.zeros((3, 256, 256)), torch.zeros((100, 5))),
            (torch.zeros((3, 256, 256)), torch.zeros((100, 5))),
            (torch.zeros((3, 256, 256)), torch.zeros((100, 5))),
        ]
        collate = DetectionCollateFN()
        _, targets = collate(samples)

        for device in self.devices:
            predictions = [
                torch.randn((5, 1, 256 // 8, 256 // 8, 4 + 1 + 10)).to(device),
                torch.randn((5, 1, 256 // 16, 256 // 16, 4 + 1 + 10)).to(device),
                torch.randn((5, 1, 256 // 32, 256 // 32, 4 + 1 + 10)).to(device),
            ]

            for loss in [
                YoloXDetectionLoss(strides=[8, 16, 32], num_classes=10, use_l1=True, iou_type="giou"),
                YoloXDetectionLoss(strides=[8, 16, 32], num_classes=10, use_l1=True, iou_type="iou"),
                YoloXDetectionLoss(strides=[8, 16, 32], num_classes=10, use_l1=False),
                YoloXFastDetectionLoss(strides=[8, 16, 32], num_classes=10, use_l1=True),
                YoloXFastDetectionLoss(strides=[8, 16, 32], num_classes=10, use_l1=False),
            ]:
                result = loss(predictions, targets.to(device))
                print(result)

    def test_yolo_x_checkpoint_solver(self):
        """
        This test checks whether we can:
        1. load an old pretrained weights for YoloX that has non-matching keys  (Using custom solver under the hood).
        2. load a regular checkpoint (As if one would train a model from scratch).
        3. that both models produce the same output.

        :return:
        """
        model_variant = [Models.YOLOX_S, Models.YOLOX_M, Models.YOLOX_L, Models.YOLOX_T, Models.YOLOX_N]
        for model_name in model_variant:
            model = models.get(model_name, pretrained_weights="coco").eval()
            input = torch.randn((1, 3, 320, 320))

            output1 = model(input)

            sd = model.state_dict()

            with tempfile.TemporaryDirectory() as tmp_dirname:
                path = os.path.join(tmp_dirname, f"{model_name}_coco.pth")
                torch.save({"net": sd}, path)
                model = models.get(model_name, num_classes=80, checkpoint_path=path).eval()
                output2 = model(input)

            assert torch.allclose(output1[0], output2[0], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
