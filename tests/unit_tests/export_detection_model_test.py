import logging
import os
import tempfile
import unittest

import cv2
import numpy as np
import onnxruntime
import torch
from torch.utils.data import DataLoader

from super_gradients.common.object_names import Models
from super_gradients.module_interfaces import ExportableObjectDetectionModel
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val  # noqa
from super_gradients.training.utils.media.image import load_image


class TestDetectionModelExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        self.test_image_path = "../data/tinycoco/images/val2017/000000444010.jpg"

    def test_the_most_common_export_use_case(self):
        """
        Test the most common export use case - export to ONNX with all default parameters
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            ppyolo_e.export(out_path, input_image_shape=(640, 640))

    def test_export_to_onnxruntime_flat(self):
        """
        Test export to ONNX with flat predictions
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            ppyolo_e.export(out_path, input_image_shape=(640, 640), engine="onnxruntime", output_predictions_format="flat")

            image = self._get_test_image()

            session = onnxruntime.InferenceSession(out_path)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: image})
            for r in result:
                print(r.shape, r.dtype, r)

            flat_predictions = result[0]
            assert flat_predictions.shape[1] == 7

    def test_export_to_onnxruntime_batch_format(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            ppyolo_e.export(
                out_path,
                engine="onnxruntime",
                input_image_shape=(640, 640),
                num_pre_nms_predictions=1000,
                max_predictions_per_image=None,
                output_predictions_format="batch",
            )

            image = self._get_test_image()

            session = onnxruntime.InferenceSession(out_path)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: image})
            for r in result:
                print(r.shape, r.dtype, r)

            num_predictions, pred_boxes, pred_scores, pred_classes = result
            assert num_predictions.shape == (1, 1)
            assert pred_boxes.shape == (1, 1000, 4)
            assert pred_scores.shape == (1, 1000)
            assert pred_classes.shape == (1, 1000)

    def test_export_model_with_custom_input_image_shape(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            ppyolo_e.export(out_path, engine="onnxruntime", input_image_shape=(320, 320), output_predictions_format="flat")

            image = self._get_test_image(image_shape=(320, 320))

            session = onnxruntime.InferenceSession(out_path)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: image})
            for r in result:
                print(r.shape, r.dtype, r)

            flat_predictions = result[0]
            assert flat_predictions.shape[1] == 7

    def test_export_model_to_tensorrt(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            ppyolo_e.export(out_path, engine="onnxruntime", input_image_shape=(640, 640), max_predictions_per_image=300)

            image = self._get_test_image()

            session = onnxruntime.InferenceSession(out_path)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: image})
            for r in result:
                print(r.shape, r.dtype, r)

            num_predictions, pred_boxes, pred_scores, pred_classes = result
            assert num_predictions.shape == (1, 1)
            assert pred_boxes.shape == (1, 300, 4)
            assert pred_scores.shape == (1, 300)
            assert pred_classes.shape == (1, 300)

    def test_export_quantized_with_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8, num_workers=0)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            ppyolo_e.export(
                out_path,
                engine="onnxruntime",
                max_predictions_per_image=300,
                input_image_shape=(640, 640),
                output_predictions_format="batch",
                quantize=True,
                calibration_loader=dummy_calibration_loader,
            )

            image = self._get_test_image()

            session = onnxruntime.InferenceSession(out_path)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: image})
            for r in result:
                print(r.shape, r.dtype, r)

            num_predictions, pred_boxes, pred_scores, pred_classes = result
            assert num_predictions.shape == (1, 1)
            assert pred_boxes.shape == (1, 300, 4)
            assert pred_scores.shape == (1, 300)
            assert pred_classes.shape == (1, 300)
            assert pred_classes.dtype == np.int64

    def test_export_quantized_with_calibration_to_tensorrt(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            ppyolo_e.export(
                out_path,
                engine="tensorrt",
                max_predictions_per_image=300,
                input_image_shape=(640, 640),
                output_predictions_format="batch",
                quantize=True,
                calibration_loader=dummy_calibration_loader,
            )

    def _get_test_image(self, image_shape=(640, 640)):
        """

        :param image_shape: Output image shape (rows, cols)
        :return: Image in NCHW format
        """

        image = load_image(self.test_image_path)
        image = cv2.resize(image, dsize=tuple(reversed(image_shape)), interpolation=cv2.INTER_LINEAR)
        image = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2)) / 255.0
        return image.astype(np.float32)


if __name__ == "__main__":
    unittest.main()
