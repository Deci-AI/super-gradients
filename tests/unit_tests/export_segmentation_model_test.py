import logging
import os
import tempfile
import unittest

import cv2
import numpy as np
import onnxruntime
import torch
from matplotlib import pyplot as plt
from super_gradients.common.object_names import Models
from super_gradients.conversion.conversion_enums import ExportQuantizationMode
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_install
from super_gradients.import_utils import import_pytorch_quantization_or_install
from super_gradients.module_interfaces import ExportableSegmentationModel, SegmentationModelExportResult
from super_gradients.training import models
from super_gradients.training.datasets.datasets_conf import CITYSCAPES_DEFAULT_SEGMENTATION_CLASSES_LIST
from super_gradients.training.utils.detection_utils import DetectionVisualization
from super_gradients.training.utils.export_utils import infer_image_shape_from_model, infer_image_input_channels
from super_gradients.training.utils.media.image import load_image
from super_gradients.training.utils.visualization.segmentation import overlay_segmentation
from torch.utils.data import DataLoader

gs = import_onnx_graphsurgeon_or_install()
import_pytorch_quantization_or_install()


class TestSegmentationModelExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        this_dir = os.path.dirname(__file__)
        self.test_image_path = os.path.join(this_dir, "../data/tinycoco/images/train2017/000000017627.jpg")

        self.models_to_test = [
            Models.DDRNET_23,
            Models.SEGFORMER_B0,
            Models.PP_LITE_T_SEG50,
            Models.STDC1_SEG50,
        ]

    def test_infer_input_image_shape_from_model(self):
        assert infer_image_shape_from_model(models.get(Models.DDRNET_23, num_classes=80, pretrained_weights=None)) is None
        assert infer_image_shape_from_model(models.get(Models.SEGFORMER_B0, num_classes=80, pretrained_weights=None)) is None
        assert infer_image_shape_from_model(models.get(Models.PP_LITE_T_SEG, num_classes=80, pretrained_weights=None)) is None

        assert infer_image_shape_from_model(models.get(Models.DDRNET_23, pretrained_weights="cityscapes")) == (1024, 2048)
        assert infer_image_shape_from_model(models.get(Models.SEGFORMER_B0, pretrained_weights="cityscapes")) == (1024, 2048)
        assert infer_image_shape_from_model(models.get(Models.PP_LITE_T_SEG50, pretrained_weights="cityscapes")) == (512, 1024)
        assert infer_image_shape_from_model(models.get(Models.STDC1_SEG50, pretrained_weights="cityscapes")) == (512, 1024)

    def test_infer_input_image_num_channels_from_model(self):
        assert infer_image_input_channels(models.get(Models.DDRNET_23, num_classes=80, pretrained_weights=None)) == 3
        assert infer_image_input_channels(models.get(Models.SEGFORMER_B0, num_classes=80, pretrained_weights=None)) == 3
        assert infer_image_input_channels(models.get(Models.PP_LITE_T_SEG50, num_classes=80, pretrained_weights=None)) == 3
        assert infer_image_input_channels(models.get(Models.STDC1_SEG50, num_classes=80, pretrained_weights=None)) == 3

        assert infer_image_input_channels(models.get(Models.DDRNET_23, pretrained_weights="cityscapes")) == 3
        assert infer_image_input_channels(models.get(Models.SEGFORMER_B0, pretrained_weights="cityscapes")) == 3
        assert infer_image_input_channels(models.get(Models.PP_LITE_T_SEG50, pretrained_weights="cityscapes")) == 3
        assert infer_image_input_channels(models.get(Models.STDC1_SEG50, pretrained_weights="cityscapes")) == 3

    def test_export_to_onnxruntime_and_run(self):
        """
        Test export to ONNX with flat predictions
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in self.models_to_test:
                with self.subTest(model_type=model_type):
                    model_name = str(model_type).lower().replace(".", "_")
                    out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_flat.onnx")

                    model_arch: ExportableSegmentationModel = models.get(model_name, pretrained_weights="cityscapes")
                    export_result = model_arch.export(
                        out_path,
                        input_image_shape=(640, 640),  # Force .export() to infer image shape from the model itself
                    )

                    [segmentation_mask] = self._run_inference_with_onnx(export_result)
                    self.assertTrue(segmentation_mask.shape[0] == 1)
                    self.assertTrue(segmentation_mask.shape[1] == 640)
                    self.assertTrue(segmentation_mask.shape[2] == 640)

    def test_export_model_with_binary_head(self):
        """
        Test export to ONNX with flat predictions
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in self.models_to_test:
                with self.subTest(model_type=model_type):
                    model_name = str(model_type).lower().replace(".", "_")
                    out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_flat.onnx")

                    model_arch: ExportableSegmentationModel = models.get(model_name, pretrained_weights="cityscapes")
                    model_arch.replace_head(new_num_classes=1)
                    export_result = model_arch.export(
                        out_path,
                        confidence_threshold=0.5,
                        input_image_shape=(640, 640),  # Force .export() to infer image shape from the model itself
                    )

                    [segmentation_mask] = self._run_inference_with_onnx(export_result)
                    self.assertTrue(np.isin(segmentation_mask, [0, 1]).all())
                    self.assertTrue(segmentation_mask.shape[0] == 1)
                    self.assertTrue(segmentation_mask.shape[1] == 640)
                    self.assertTrue(segmentation_mask.shape[2] == 640)

    def test_export_int8_quantized_with_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in self.models_to_test:
                with self.subTest(model_type=model_type):
                    model_name = str(model_type).lower().replace(".", "_")
                    out_path = os.path.join(tmpdirname, f"{model_name}.onnx")

                    dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
                    dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

                    model_arch: ExportableSegmentationModel = models.get(model_name, pretrained_weights="cityscapes")
                    export_result = model_arch.export(
                        out_path,
                        input_image_shape=(640, 640),  # Force .export() to infer image shape from the model itself
                        quantization_mode=ExportQuantizationMode.INT8,
                        calibration_loader=dummy_calibration_loader,
                    )

                    [segmentation_mask] = self._run_inference_with_onnx(export_result)
                    self.assertTrue(segmentation_mask.shape[0] == 1)
                    self.assertTrue(segmentation_mask.shape[1] == 640)
                    self.assertTrue(segmentation_mask.shape[2] == 640)

    def _run_inference_with_onnx(self, export_result: SegmentationModelExportResult):
        # onnx_filename = out_path, input_shape = export_result.image_shape, output_predictions_format = output_predictions_format

        image = self._get_image_as_bchw(export_result.input_image_shape)
        image_8u = self._get_image(export_result.input_image_shape)

        session = onnxruntime.InferenceSession(export_result.output)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: image})

        class_names = CITYSCAPES_DEFAULT_SEGMENTATION_CLASSES_LIST
        color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

        segmentation_mask = result[0][0]  # [H, W]
        overlay = overlay_segmentation(
            pred_mask=segmentation_mask, image=image_8u, alpha=0.5, num_classes=len(class_names), colors=color_mapping, class_names=class_names
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(os.path.basename(export_result.output))
        plt.tight_layout()
        plt.show()

        return result

    def test_export_already_quantized_model(self):
        from super_gradients.training.utils.quantization import SelectiveQuantizer

        for model_type in self.models_to_test:
            with self.subTest(model_type=model_type):
                model = models.get(model_type, pretrained_weights="cityscapes")
                q_util = SelectiveQuantizer(
                    default_quant_modules_calibrator_weights="max",
                    default_quant_modules_calibrator_inputs="histogram",
                    default_per_channel_quant_weights=True,
                    default_learn_amax=False,
                    verbose=True,
                )
                q_util.quantize_module(model)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    output_model1 = os.path.join(tmpdirname, f"{model_type}_quantized_explicit_int8.onnx")
                    output_model2 = os.path.join(tmpdirname, f"{model_type}_quantized.onnx")

                    # If model is already quantized to int8, the export should be successful but model should not be quantized again
                    model.export(
                        output_model1,
                        quantization_mode=ExportQuantizationMode.INT8,
                    )

                    # If model is quantized but quantization mode is not specified, the export should be also successful
                    # but model should not be quantized again
                    model.export(
                        output_model2,
                        quantization_mode=None,
                    )

                    # If model is already quantized to int8, we should not be able to export model to FP16
                    with self.assertRaises(RuntimeError):
                        model.export(
                            "yolo_nas_s_quantized.onnx",
                            quantization_mode=ExportQuantizationMode.FP16,
                        )

    def _get_image_as_bchw(self, image_shape=(640, 640)):
        """

        :param image_shape: Output image shape (rows, cols)
        :return: Image in NCHW format
        """

        image = load_image(self.test_image_path)
        image = cv2.resize(image, dsize=tuple(reversed(image_shape)), interpolation=cv2.INTER_LINEAR)
        image = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))
        return image

    def _get_image(self, image_shape=(640, 640)):
        """

        :param image_shape: Output image shape (rows, cols)
        :return: Image in HWC format
        """

        image = load_image(self.test_image_path)
        image = cv2.resize(image, dsize=tuple(reversed(image_shape)), interpolation=cv2.INTER_LINEAR)
        return image


if __name__ == "__main__":
    unittest.main()
