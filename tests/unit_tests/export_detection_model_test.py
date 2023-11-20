import logging
import os
import random
import tempfile
import unittest

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from matplotlib import pyplot as plt

from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_fail_with_instructions
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer
from torch import nn
from torch.utils.data import DataLoader

from super_gradients.common.object_names import Models
from super_gradients.conversion.conversion_enums import ExportTargetBackend, ExportQuantizationMode, DetectionOutputFormatMode
from super_gradients.conversion.onnx.nms import PickNMSPredictionsAndReturnAsFlatResult, PickNMSPredictionsAndReturnAsBatchedResult
from super_gradients.conversion.tensorrt.nms import ConvertTRTFormatToFlatTensor
from super_gradients.module_interfaces import ExportableObjectDetectionModel
from super_gradients.module_interfaces.exportable_detector import ModelExportResult
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val  # noqa
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.utils.detection_utils import DetectionVisualization
from super_gradients.training.utils.export_utils import infer_image_shape_from_model, infer_image_input_channels
from super_gradients.training.utils.media.image import load_image


gs = import_onnx_graphsurgeon_or_fail_with_instructions()


class TestDetectionModelExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        this_dir = os.path.dirname(__file__)
        self.test_image_path = os.path.join(this_dir, "../data/tinycoco/images/val2017/000000444010.jpg")

    def test_export_model_on_small_size(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_S,
                Models.PP_YOLOE_S,
                Models.YOLOX_S,
            ]:
                out_path = os.path.join(tmpdirname, model_type + ".onnx")
                ppyolo_e: ExportableObjectDetectionModel = models.get(model_type, pretrained_weights="coco")
                result = ppyolo_e.export(
                    out_path,
                    input_image_shape=(64, 64),
                    num_pre_nms_predictions=2000,
                    max_predictions_per_image=1000,
                    output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                )
                assert result.input_image_dtype == torch.uint8
                assert result.input_image_shape == (64, 64)

    def test_the_most_common_export_use_case(self):
        """
        Test the most common export use case - export to ONNX with all default parameters
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            result = ppyolo_e.export(out_path)
            assert result.input_image_dtype == torch.uint8
            assert result.input_image_shape == (640, 640)
            assert result.input_image_channels == 3

    def test_models_produce_half(self):
        if not torch.cuda.is_available():
            self.skipTest("This test was skipped because target machine has not CUDA devices")

        input = torch.randn(1, 3, 640, 640).half().cuda()

        model = models.get(Models.YOLO_NAS_S, num_classes=80, pretrained_weights=None)
        model = nn.Sequential(model, model.get_decoding_module(100)).cuda().eval().half()
        output = model(input)
        assert output[0].dtype == torch.float16
        assert output[1].dtype == torch.float16

        model = models.get(Models.PP_YOLOE_S, num_classes=80, pretrained_weights=None)
        model = nn.Sequential(model, model.get_decoding_module(100)).cuda().eval().half()
        output = model(input)
        assert output[0].dtype == torch.float16
        assert output[1].dtype == torch.float16

        model = models.get(Models.YOLOX_S, num_classes=80, pretrained_weights=None)
        model = nn.Sequential(model, model.get_decoding_module(100)).cuda().eval().half()
        output = model(input)
        assert output[0].dtype == torch.float16
        assert output[1].dtype == torch.float16

    def test_infer_input_image_shape_from_model(self):
        assert infer_image_shape_from_model(models.get(Models.PP_YOLOE_S, num_classes=80, pretrained_weights=None)) is None
        assert infer_image_shape_from_model(models.get(Models.YOLO_NAS_S, num_classes=80, pretrained_weights=None)) is None
        assert infer_image_shape_from_model(models.get(Models.YOLOX_S, num_classes=80, pretrained_weights=None)) is None

        assert infer_image_shape_from_model(models.get(Models.PP_YOLOE_S, pretrained_weights="coco")) == (640, 640)
        assert infer_image_shape_from_model(models.get(Models.YOLO_NAS_S, pretrained_weights="coco")) == (640, 640)
        assert infer_image_shape_from_model(models.get(Models.YOLOX_S, pretrained_weights="coco")) == (640, 640)

    def test_infer_input_image_num_channels_from_model(self):
        assert infer_image_input_channels(models.get(Models.PP_YOLOE_S, num_classes=80, pretrained_weights=None)) == 3
        assert infer_image_input_channels(models.get(Models.YOLO_NAS_S, num_classes=80, pretrained_weights=None)) == 3
        assert infer_image_input_channels(models.get(Models.YOLOX_S, num_classes=80, pretrained_weights=None)) == 3

        assert infer_image_input_channels(models.get(Models.PP_YOLOE_S, pretrained_weights="coco")) == 3
        assert infer_image_input_channels(models.get(Models.YOLO_NAS_S, pretrained_weights="coco")) == 3
        assert infer_image_input_channels(models.get(Models.YOLOX_S, pretrained_weights="coco")) == 3

    def test_export_to_onnxruntime_flat(self):
        """
        Test export to ONNX with flat predictions
        """
        output_predictions_format = DetectionOutputFormatMode.FLAT_FORMAT
        confidence_threshold = 0.7
        nms_threshold = 0.6

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_S,
                Models.PP_YOLOE_S,
                Models.YOLOX_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_flat.onnx")

                model_arch: ExportableObjectDetectionModel = models.get(model_name, pretrained_weights="coco")
                export_result = model_arch.export(
                    out_path,
                    input_image_shape=None,  # Force .export() to infer image shape from the model itself
                    engine=ExportTargetBackend.ONNXRUNTIME,
                    output_predictions_format=output_predictions_format,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                )

                [flat_predictions] = self._run_inference_with_onnx(export_result)

                # Check that all predictions have confidence >= confidence_threshold
                assert (flat_predictions[:, 5] >= confidence_threshold).all()

    def test_export_to_onnxruntime_batch_format(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.7
        nms_threshold = 0.6
        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_S,
                Models.PP_YOLOE_S,
                Models.YOLOX_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_batch.onnx")

                model_arch: ExportableObjectDetectionModel = models.get(model_name, pretrained_weights="coco")
                export_result = model_arch.export(
                    out_path,
                    input_image_shape=None,  # Force .export() to infer image shape from the model itself
                    engine=ExportTargetBackend.ONNXRUNTIME,
                    output_predictions_format=output_predictions_format,
                    nms_threshold=nms_threshold,
                    confidence_threshold=confidence_threshold,
                )

                self._run_inference_with_onnx(export_result)

    def test_export_to_tensorrt_flat(self):
        """
        Test export to tensorrt with flat predictions
        """
        output_predictions_format = DetectionOutputFormatMode.FLAT_FORMAT
        confidence_threshold = 0.7

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_S,
                Models.PP_YOLOE_S,
                Models.YOLOX_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_flat.onnx")

                model_arch: ExportableObjectDetectionModel = models.get(model_name, pretrained_weights="coco")
                export_result = model_arch.export(
                    out_path,
                    input_image_shape=None,  # Force .export() to infer image shape from the model itself
                    engine=ExportTargetBackend.TENSORRT,
                    output_predictions_format=output_predictions_format,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=0.6,
                )
                assert export_result is not None

    def test_export_to_tensorrt_batch_format(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.25
        nms_threshold = 0.6
        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_S,
                Models.PP_YOLOE_S,
                Models.YOLOX_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_batch.onnx")

                model_arch: ExportableObjectDetectionModel = models.get(model_name, pretrained_weights="coco")
                export_result = model_arch.export(
                    out_path,
                    input_image_shape=None,  # Force .export() to infer image shape from the model itself
                    engine=ExportTargetBackend.TENSORRT,
                    output_predictions_format=output_predictions_format,
                    nms_threshold=nms_threshold,
                    confidence_threshold=confidence_threshold,
                )
                assert export_result is not None

    def test_export_to_tensorrt_batch_format_yolox_s(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.25
        nms_threshold = 0.6
        model_type = Models.YOLOX_S
        device = "cpu"

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_name = str(model_type).lower().replace(".", "_")
            out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_batch.onnx")

            model_arch: ExportableObjectDetectionModel = models.get(model_name, pretrained_weights="coco")
            export_result = model_arch.export(
                out_path,
                input_image_shape=None,  # Force .export() to infer image shape from the model itself
                device=device,
                engine=ExportTargetBackend.TENSORRT,
                output_predictions_format=output_predictions_format,
                nms_threshold=nms_threshold,
                confidence_threshold=confidence_threshold,
            )
            assert export_result is not None

    def test_export_to_tensorrt_batch_format_yolo_nas_s(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.25
        nms_threshold = 0.6
        model_type = Models.YOLO_NAS_S
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_name = str(model_type).lower().replace(".", "_")
            out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_batch.onnx")

            model_arch: ExportableObjectDetectionModel = models.get(model_name, pretrained_weights="coco")
            export_result = model_arch.export(
                out_path,
                input_image_shape=None,  # Force .export() to infer image shape from the model itself
                engine=ExportTargetBackend.TENSORRT,
                output_predictions_format=output_predictions_format,
                nms_threshold=nms_threshold,
                confidence_threshold=confidence_threshold,
            )
            assert export_result is not None

    def test_export_to_tensorrt_batch_format_ppyolo_e(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.25
        nms_threshold = 0.6
        model_type = Models.PP_YOLOE_S
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_name = str(model_type).lower().replace(".", "_")
            out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_batch.onnx")

            model_arch: ExportableObjectDetectionModel = models.get(model_name, pretrained_weights="coco")
            export_result = model_arch.export(
                out_path,
                input_image_shape=None,  # Force .export() to infer image shape from the model itself
                engine=ExportTargetBackend.TENSORRT,
                output_predictions_format=output_predictions_format,
                nms_threshold=nms_threshold,
                confidence_threshold=confidence_threshold,
            )
            assert export_result is not None

    def test_export_model_with_custom_input_image_shape(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s_custom_image_shape.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(out_path, engine=ExportTargetBackend.ONNXRUNTIME, input_image_shape=(320, 320), output_predictions_format="flat")
            [flat_predictions] = self._run_inference_with_onnx(export_result)

            assert flat_predictions.shape[1] == 7

    def test_export_with_fp16_quantization(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            self.skipTest("No CUDA or MPS device available")

        max_predictions_per_image = 300
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s_with_fp16_quantization.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(
                out_path,
                device=device,
                engine=ExportTargetBackend.ONNXRUNTIME,
                max_predictions_per_image=max_predictions_per_image,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.FP16,
            )

            num_predictions, pred_boxes, pred_scores, pred_classes = self._run_inference_with_onnx(export_result)

            assert num_predictions.shape == (1, 1)
            assert pred_boxes.shape == (1, max_predictions_per_image, 4)
            assert pred_scores.shape == (1, max_predictions_per_image)
            assert pred_classes.shape == (1, max_predictions_per_image)
            assert pred_classes.dtype == np.int64

    def test_export_with_fp16_quantization_tensort_from_cpu(self):
        """
        This test checks that we can export model with FP16 quantization.
        It requires CUDA and moves model to CUDA device under the hood.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA device is required for this test")

        max_predictions_per_image = 300
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s_with_fp16_quantization.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(
                out_path,
                engine=ExportTargetBackend.TENSORRT,
                max_predictions_per_image=max_predictions_per_image,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.FP16,
            )
            assert export_result is not None

    def test_export_with_fp16_quantization_tensort(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            self.skipTest("No CUDA or MPS device available")

        max_predictions_per_image = 300
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s_with_fp16_quantization.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(
                out_path,
                device=device,
                engine=ExportTargetBackend.TENSORRT,
                max_predictions_per_image=max_predictions_per_image,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.FP16,
            )
            assert export_result is not None

    def test_export_with_int8_quantization(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ppyoloe_s_with_int8_quantization.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8, num_workers=0)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(
                out_path,
                engine=ExportTargetBackend.ONNXRUNTIME,
                max_predictions_per_image=300,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.INT8,
                calibration_loader=dummy_calibration_loader,
            )

            num_predictions, pred_boxes, pred_scores, pred_classes = self._run_inference_with_onnx(export_result)

            assert num_predictions.shape == (1, 1)
            assert pred_boxes.shape == (1, 300, 4)
            assert pred_scores.shape == (1, 300)
            assert pred_classes.shape == (1, 300)
            assert pred_classes.dtype == np.int64

    def test_export_quantized_with_calibration_to_tensorrt(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "pp_yoloe_s_quantized_with_calibration.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(
                out_path,
                engine=ExportTargetBackend.TENSORRT,
                max_predictions_per_image=300,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.INT8,
                calibration_loader=dummy_calibration_loader,
            )
            assert export_result is not None

    def test_export_yolonas_quantized_with_calibration_to_tensorrt(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "yolonas_s_quantized_with_calibration.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(
                out_path,
                engine=ExportTargetBackend.TENSORRT,
                num_pre_nms_predictions=300,
                max_predictions_per_image=100,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.INT8,
                calibration_loader=dummy_calibration_loader,
            )
            assert export_result is not None

    def test_export_yolox_quantized_int8_with_calibration_to_tensorrt(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "yolox_quantized_with_calibration.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.YOLOX_S, pretrained_weights="coco")
            export_result = ppyolo_e.export(
                out_path,
                engine=ExportTargetBackend.TENSORRT,
                num_pre_nms_predictions=300,
                max_predictions_per_image=100,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.INT8,
                calibration_loader=dummy_calibration_loader,
            )
            assert export_result is not None

    def _run_inference_with_onnx(self, export_result: ModelExportResult):
        # onnx_filename = out_path, input_shape = export_result.image_shape, output_predictions_format = output_predictions_format

        image = self._get_image_as_bchw(export_result.input_image_shape)
        image_8u = self._get_image(export_result.input_image_shape)

        session = onnxruntime.InferenceSession(export_result.output)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: image})

        class_names = COCO_DETECTION_CLASSES_LIST
        color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

        if export_result.output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
            flat_predictions = result[0]  # [N, (batch_index, x1, y1, x2, y2, score, class]
            assert flat_predictions.shape[1] == 7

            for i in range(flat_predictions.shape[0]):
                x1, y1, x2, y2 = flat_predictions[i, 1:5]
                class_score = flat_predictions[i, 5]
                class_label = int(flat_predictions[i, 6])

                image_8u = DetectionVisualization.draw_box_title(
                    image_np=image_8u,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=class_label,
                    class_names=class_names,
                    color_mapping=color_mapping,
                    box_thickness=2,
                    pred_conf=class_score,
                )
        else:
            num_predictions, pred_boxes, pred_scores, pred_classes = result
            for pred_index in range(num_predictions[0, 0]):
                x1, y1, x2, y2 = pred_boxes[0, pred_index]
                class_score = pred_scores[0, pred_index]
                class_label = pred_classes[0, pred_index]

                image_8u = DetectionVisualization.draw_box_title(
                    image_np=image_8u,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=class_label,
                    class_names=class_names,
                    color_mapping=color_mapping,
                    box_thickness=2,
                    pred_conf=class_score,
                )

        plt.figure(figsize=(10, 10))
        plt.imshow(image_8u)
        plt.title(os.path.basename(export_result.output))
        plt.tight_layout()
        plt.show()

        return result

    def test_export_already_quantized_model(self):
        model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights="max",
            default_quant_modules_calibrator_inputs="histogram",
            default_per_channel_quant_weights=True,
            default_learn_amax=False,
            verbose=True,
        )
        q_util.quantize_module(model)

        with tempfile.TemporaryDirectory() as tmpdirname:
            output_model1 = os.path.join(tmpdirname, "yolo_nas_s_quantized_explicit_int8.onnx")
            output_model2 = os.path.join(tmpdirname, "yolo_nas_s_quantized.onnx")

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

            # Assert two files are the same
            # with open(output_model1, "rb") as f1, open(output_model2, "rb") as f2:
            #     assert hashlib.md5(f1.read()) == hashlib.md5(f2.read())

    def manual_test_export_export_all_variants(self):
        """
        This test is not run automatically, it is used to generate all possible export variants of the model
        for benchmarking purposes.
        """
        export_dir = "export_all_variants"
        os.makedirs(export_dir, exist_ok=True)

        benchmark_command_dir = "benchmark_command.sh"
        with open(benchmark_command_dir, "w") as f:
            pass

        for output_predictions_format in [DetectionOutputFormatMode.BATCH_FORMAT, DetectionOutputFormatMode.FLAT_FORMAT]:
            for engine in [ExportTargetBackend.ONNXRUNTIME, ExportTargetBackend.TENSORRT]:
                for quantization in [None, ExportQuantizationMode.FP16, ExportQuantizationMode.INT8]:
                    device = "cpu"
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif torch.backends.mps.is_available() and quantization == ExportQuantizationMode.FP16:
                        # Skip this case because when using MPS device we are getting:
                        # RuntimeError: Placeholder storage has not been allocated on MPS device!
                        # And when using CPU:
                        # RuntimeError: RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
                        continue

                    # if quantization == ExportQuantizationMode.FP16 and device == "cpu":
                    #     # Skip this case because the FP16 quantization uses model inference
                    #     pass

                    for model_type in [
                        Models.YOLOX_S,
                        Models.PP_YOLOE_S,
                        Models.YOLO_NAS_S,
                    ]:
                        model_name = str(model_type).lower()
                        model = models.get(model_type, pretrained_weights="coco")
                        quantization_suffix = f"_{quantization.value}" if quantization is not None else ""
                        onnx_filename = f"{model_name}_{engine.value}_{output_predictions_format.value}{quantization_suffix}.onnx"

                        with self.subTest(msg=onnx_filename):
                            model.export(
                                os.path.join(export_dir, onnx_filename),
                                device=device,
                                quantization_mode=quantization,
                                engine=engine,
                                output_predictions_format=output_predictions_format,
                                preprocessing=False,
                                postprocessing=False,
                            )

                            with open(benchmark_command_dir, "a") as f:
                                quantization_param = "--int8" if quantization == ExportQuantizationMode.INT8 else "--fp16"
                                output_file_log = onnx_filename.replace(".onnx", ".log")
                                trtexec_command = (
                                    f"/usr/src/tensorrt/bin/trtexec "
                                    f"--onnx=/deci/eugene/{onnx_filename} {quantization_param} "
                                    f"--avgRuns=100 --duration=15 > /deci/eugene/{output_file_log}\n"
                                )
                                f.write(trtexec_command)

    def test_trt_nms_convert_to_flat_result(self):
        batch_size = 7
        max_predictions_per_image = 100

        if torch.cuda.is_available():
            available_devices = ["cpu", "cuda"]
            available_dtypes = [torch.float16, torch.float32]
        else:
            available_devices = ["cpu"]
            available_dtypes = [torch.float32]

        for num_predictions_max in [0, max_predictions_per_image // 2, max_predictions_per_image]:
            for device in available_devices:
                for dtype in available_dtypes:
                    num_detections = torch.randint(0, num_predictions_max + 1, (batch_size, 1), dtype=torch.int32)
                    detection_boxes = torch.randn((batch_size, max_predictions_per_image, 4), dtype=dtype)
                    detection_scores = torch.randn((batch_size, max_predictions_per_image)).sigmoid().to(dtype)
                    detection_classes = torch.randint(0, 80, (batch_size, max_predictions_per_image), dtype=torch.int32)

                    torch_module = ConvertTRTFormatToFlatTensor(batch_size, max_predictions_per_image)
                    flat_predictions_torch = torch_module(num_detections, detection_boxes, detection_scores, detection_classes)
                    print(flat_predictions_torch.shape, flat_predictions_torch.dtype, flat_predictions_torch)

                    onnx_file = "ConvertTRTFormatToFlatTensor.onnx"

                    graph = ConvertTRTFormatToFlatTensor.as_graph(
                        batch_size=batch_size, max_predictions_per_image=max_predictions_per_image, dtype=dtype, device=device
                    )
                    model = gs.export_onnx(graph)
                    onnx.checker.check_model(model)
                    onnx.save(model, onnx_file)

                    session = onnxruntime.InferenceSession(onnx_file)

                    inputs = [o.name for o in session.get_inputs()]
                    outputs = [o.name for o in session.get_outputs()]

                    [flat_predictions_onnx] = session.run(
                        output_names=outputs,
                        input_feed={
                            inputs[0]: num_detections.numpy(),
                            inputs[1]: detection_boxes.numpy(),
                            inputs[2]: detection_scores.numpy(),
                            inputs[3]: detection_classes.numpy(),
                        },
                    )

                    np.testing.assert_allclose(flat_predictions_torch.numpy(), flat_predictions_onnx, rtol=1e-3, atol=1e-3)

    def test_onnx_nms_flat_result(self):
        num_pre_nms_predictions = 1024
        max_predictions_per_image = 128
        batch_size = 7

        if torch.cuda.is_available():
            available_devices = ["cpu", "cuda"]
            available_dtypes = [torch.float16, torch.float32]
        else:
            available_devices = ["cpu"]
            available_dtypes = [torch.float32]

        for max_detections in [0, num_pre_nms_predictions // 2, num_pre_nms_predictions, num_pre_nms_predictions * 2]:
            for device in available_devices:
                for dtype in available_dtypes:
                    # Run a few tests to ensure ONNX model produces the same results as the PyTorch model
                    # And also can handle dynamic shapes input
                    pred_boxes = torch.randn((batch_size, num_pre_nms_predictions, 4), dtype=dtype)
                    pred_scores = torch.randn((batch_size, num_pre_nms_predictions, 40), dtype=dtype)

                    selected_indexes = []
                    for batch_index in range(batch_size):
                        # num_detections = random.randrange(0, max_detections) if max_detections > 0 else 0
                        num_detections = max_detections
                        for _ in range(num_detections):
                            selected_indexes.append([batch_index, random.randrange(0, 40), random.randrange(0, num_pre_nms_predictions)])
                    selected_indexes = torch.tensor(selected_indexes, dtype=torch.int64).view(-1, 3)

                    torch_module = PickNMSPredictionsAndReturnAsFlatResult(
                        batch_size=batch_size, num_pre_nms_predictions=num_pre_nms_predictions, max_predictions_per_image=max_predictions_per_image
                    )
                    torch_result = torch_module(pred_boxes, pred_scores, selected_indexes)

                    with tempfile.TemporaryDirectory() as temp_dir:
                        onnx_file = os.path.join(temp_dir, "PickNMSPredictionsAndReturnAsFlatResult.onnx")
                        graph = PickNMSPredictionsAndReturnAsFlatResult.as_graph(
                            batch_size=batch_size,
                            num_pre_nms_predictions=num_pre_nms_predictions,
                            max_predictions_per_image=max_predictions_per_image,
                            device=device,
                            dtype=dtype,
                        )

                        model = gs.export_onnx(graph)
                        onnx.checker.check_model(model)
                        onnx.save(model, onnx_file)

                        session = onnxruntime.InferenceSession(onnx_file)

                        inputs = [o.name for o in session.get_inputs()]
                        outputs = [o.name for o in session.get_outputs()]

                        [onnx_result] = session.run(
                            outputs, {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: selected_indexes.numpy()}
                        )

                        np.testing.assert_allclose(torch_result.numpy(), onnx_result, rtol=1e-3, atol=1e-3)

    def test_onnx_nms_batch_result(self):
        num_pre_nms_predictions = 1024
        max_predictions_per_image = 128
        batch_size = 7

        if torch.cuda.is_available():
            available_devices = ["cpu", "cuda"]
            available_dtypes = [torch.float16, torch.float32]
        else:
            available_devices = ["cpu"]
            available_dtypes = [torch.float32]

        for max_detections in [0, num_pre_nms_predictions // 2, num_pre_nms_predictions, num_pre_nms_predictions * 2]:
            for device in available_devices:
                for dtype in available_dtypes:
                    # Run a few tests to ensure ONNX model produces the same results as the PyTorch model
                    # And also can handle dynamic shapes input
                    pred_boxes = torch.randn((batch_size, num_pre_nms_predictions, 4), dtype=dtype)
                    pred_scores = torch.randn((batch_size, num_pre_nms_predictions, 40), dtype=dtype)

                    selected_indexes = []
                    for batch_index in range(batch_size):
                        # num_detections = random.randrange(0, max_detections) if max_detections > 0 else 0
                        num_detections = max_detections
                        for _ in range(num_detections):
                            selected_indexes.append([batch_index, random.randrange(0, 40), random.randrange(0, num_pre_nms_predictions)])
                    selected_indexes = torch.tensor(selected_indexes, dtype=torch.int64).view(-1, 3)

                    torch_module = PickNMSPredictionsAndReturnAsBatchedResult(
                        batch_size=batch_size, num_pre_nms_predictions=num_pre_nms_predictions, max_predictions_per_image=max_predictions_per_image
                    )
                    torch_result = torch_module(pred_boxes, pred_scores, selected_indexes)

                    with tempfile.TemporaryDirectory() as temp_dir:
                        onnx_file = os.path.join(temp_dir, "PickNMSPredictionsAndReturnAsBatchedResult.onnx")
                        graph = PickNMSPredictionsAndReturnAsBatchedResult.as_graph(
                            batch_size=batch_size,
                            num_pre_nms_predictions=num_pre_nms_predictions,
                            max_predictions_per_image=max_predictions_per_image,
                            device=device,
                            dtype=dtype,
                        )

                        model = gs.export_onnx(graph)
                        onnx.checker.check_model(model)
                        onnx.save(model, onnx_file)

                        session = onnxruntime.InferenceSession(onnx_file)

                        inputs = [o.name for o in session.get_inputs()]
                        outputs = [o.name for o in session.get_outputs()]

                        onnx_result = session.run(outputs, {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: selected_indexes.numpy()})

                        np.testing.assert_allclose(torch_result[0].numpy(), onnx_result[0], rtol=1e-3, atol=1e-3)
                        np.testing.assert_allclose(torch_result[1].numpy(), onnx_result[1], rtol=1e-3, atol=1e-3)
                        np.testing.assert_allclose(torch_result[2].numpy(), onnx_result[2], rtol=1e-3, atol=1e-3)
                        np.testing.assert_allclose(torch_result[3].numpy(), onnx_result[3], rtol=1e-3, atol=1e-3)

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
