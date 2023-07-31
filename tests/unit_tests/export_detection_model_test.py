import logging
import os
import tempfile
import unittest

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from matplotlib import pyplot as plt
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
import onnx_graphsurgeon as gs


class TestDetectionModelExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        self.test_image_path = "../data/tinycoco/images/val2017/000000444010.jpg"
        try:
            from decibenchmark.api.client_manager import ClientManager  # noqa

            self.decibenchmark_available = True
        except ImportError:
            self.decibenchmark_available = False

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
                self._benchmark_onnx(export_result)

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
                self._benchmark_onnx(export_result)

    def test_export_to_tensorrt_batch_format_yolox_s(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.25
        nms_threshold = 0.6
        model_type = Models.YOLOX_S
        device = "cuda"

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
            self._benchmark_onnx(export_result)

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
            self._benchmark_onnx(export_result)

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
            self._benchmark_onnx(export_result)

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
            tmpdirname = "."
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

            self._benchmark_onnx(export_result)

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
            self._benchmark_onnx(export_result)

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
            self._benchmark_onnx(export_result)

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
            self._benchmark_onnx(export_result)

    def test_export_yolox_quantized_fp16(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = "."
            out_path = os.path.join(tmpdirname, "yolox_quantized_fp16_onnxruntime.onnx")

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.YOLOX_S, num_classes=80)
            export_result = ppyolo_e.export(
                out_path,
                device="cuda",
                preprocessing=False,
                postprocessing=False,
                engine=ExportTargetBackend.ONNXRUNTIME,
                num_pre_nms_predictions=300,
                max_predictions_per_image=100,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.FP16,
                onnx_export_kwargs=dict(verbose=True, do_constant_folding=True),
            )

            self._benchmark_onnx(export_result)

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

            self._benchmark_onnx(export_result)

    def test_export_ssd_mobilenet_v1_quantized_with_calibration_to_tensorrt(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ssd_mobilenet_v1.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.SSD_MOBILENET_V1, pretrained_weights="coco")
            ppyolo_e.export(
                out_path,
                engine=ExportTargetBackend.TENSORRT,
                num_pre_nms_predictions=300,
                max_predictions_per_image=100,
                nms_threshold=0.5,
                confidence_threshold=0.5,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.INT8,
                calibration_loader=dummy_calibration_loader,
            )

    def test_export_ssd_lite_mobilenet_v2_quantized_with_calibration_to_tensorrt(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "ssd_lite_mobilenet_v2.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

            ppyolo_e: ExportableObjectDetectionModel = models.get(Models.SSD_LITE_MOBILENET_V2, pretrained_weights="coco")
            ppyolo_e.export(
                out_path,
                engine=ExportTargetBackend.TENSORRT,
                num_pre_nms_predictions=300,
                nms_threshold=0.6,
                confidence_threshold=0.1,
                max_predictions_per_image=100,
                input_image_shape=(640, 640),
                output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                quantization_mode=ExportQuantizationMode.INT8,
                calibration_loader=dummy_calibration_loader,
            )

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

                image_8u = DetectionVisualization._draw_box_title(
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

                image_8u = DetectionVisualization._draw_box_title(
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

    def test_export_export_all_variants(self):
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
                        # Models.PP_YOLOE_S,
                        # Models.YOLO_NAS_S
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

        for device in ["cpu", "cuda"]:
            for dtype in [torch.float16, torch.float32]:

                num_detections = torch.randint(1, max_predictions_per_image, (batch_size, 1), dtype=torch.int32)
                detection_boxes = torch.randn((batch_size, max_predictions_per_image, 4), dtype=dtype)
                detection_scores = torch.randn((batch_size, max_predictions_per_image), dtype=dtype)
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
        onnx_file = "PickNMSPredictionsAndReturnAsFlatResult.onnx"
        graph = PickNMSPredictionsAndReturnAsFlatResult.as_graph(batch_size=7)
        model = gs.export_onnx(graph)
        onnx.checker.check_model(model)
        onnx.save(model, onnx_file)

        torch_module = PickNMSPredictionsAndReturnAsFlatResult()

        session = onnxruntime.InferenceSession(onnx_file)

        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]

        # Run a few tests to ensure ONNX model produces the same results as the PyTorch model
        # And also can handle dynamic shapes input
        pred_boxes = torch.randn((7, 800, 4), dtype=torch.float32)
        pred_scores = torch.randn((7, 800, 40), dtype=torch.float32)
        selected_indexes = torch.tensor([[6, 10, 4], [1, 13, 4], [2, 17, 2], [2, 18, 2]], dtype=torch.int64)

        torch_result = torch_module(pred_boxes, pred_scores, selected_indexes)
        onnx_result = session.run(outputs, {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: selected_indexes.numpy()})
        for r in onnx_result:
            print(r.shape, r.dtype, r)

        # Test on empty NMS result
        pred_boxes = torch.randn((7, 800, 4), dtype=torch.float32)
        pred_scores = torch.randn((7, 800, 40), dtype=torch.float32)
        selected_indexes = torch.zeros((0, 3), dtype=torch.int64)

        torch_result = torch_module(pred_boxes, pred_scores, selected_indexes)  # noqa
        onnx_result = session.run(outputs, {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: selected_indexes.numpy()})
        for r in onnx_result:
            print(r.shape, r.dtype, r)

    def test_onnx_nms_batched_result(self):
        onnx_file = "PickNMSPredictionsAndReturnAsBatchedResult.onnx"
        graph = PickNMSPredictionsAndReturnAsBatchedResult.as_graph(batch_size=7, max_predictions_per_image=100)
        model = gs.export_onnx(graph)
        onnx.checker.check_model(model)
        onnx.save(model, onnx_file)

        torch_module = PickNMSPredictionsAndReturnAsBatchedResult(batch_size=7, max_predictions_per_image=100)

        session = onnxruntime.InferenceSession(onnx_file)

        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]

        # Run a few tests to ensure ONNX model produces the same results as the PyTorch model
        # And also can handle dynamic shapes input
        pred_boxes = torch.randn((7, 800, 4), dtype=torch.float32)
        pred_scores = torch.randn((7, 800, 40), dtype=torch.float32)
        selected_indexes = torch.tensor([[6, 10, 4], [1, 13, 4], [2, 17, 2], [2, 18, 2]], dtype=torch.int64)

        torch_result = torch_module(pred_boxes, pred_scores, selected_indexes)
        onnx_result = session.run(outputs, {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: selected_indexes.numpy()})
        for r in onnx_result:
            print(r.shape, r.dtype, r)

        # Test on empty NMS result
        pred_boxes = torch.randn((7, 800, 4), dtype=torch.float32)
        pred_scores = torch.randn((7, 800, 40), dtype=torch.float32)
        selected_indexes = torch.zeros((0, 3), dtype=torch.int64)

        torch_result = torch_module(pred_boxes, pred_scores, selected_indexes)  # noqa
        onnx_result = session.run(outputs, {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: selected_indexes.numpy()})
        for r in onnx_result:
            print(r.shape, r.dtype, r)

    def _benchmark_onnx(self, export_result: ModelExportResult):
        if not self.decibenchmark_available:
            print("DeciBenchMark is not available, skipping benchmark")
            return

        from decibenchmark.api.client_manager import ClientManager
        from decibenchmark.api.hardware.jetson.jetson_device_filter import JetsonDeviceFilter
        from decibenchmark.api.hardware.nvidia_gpu.nvidia_gpu_device_filter import NvidiaGpuDeviceFilter  # noqa
        from decibenchmark.common.hardware.nvidia_gpu.nvidia_gpu_model import NvidiaGpuModel  # noqa

        from decibenchmark.common.hardware.jetson.jetson_model import JetsonModel
        from decibenchmark.common.execmethod.trt_exec_params import TrtExecParams

        # from decibenchmark.common.hardware.nvidia_gpu.nvidia_gpu_model import NvidiaGpuModel

        # Create client manager
        client_manager = ClientManager.create()

        # Get jetson client
        client = client_manager.jetson

        precision = "--int8" if export_result.quantization_mode == ExportQuantizationMode.INT8 else "--fp16"

        job = client.benchmark.trt_exec(
            # JetsonDeviceFilter(jetson_model=JetsonModel.XAVIER_NX),
            JetsonDeviceFilter(jetson_model=JetsonModel.AGX_ORIN),
            # NvidiaGpuDeviceFilter(nvidia_gpu_model=NvidiaGpuModel.RTX_A5000),
            TrtExecParams(extra_cmd_params=[precision, "--avgRuns=100", "--duration=15"]),
        ).dispatch(export_result.output)

        benchmark_result = job.wait_for_result(timeout=-1)

        # Get the latency and throughput
        print(f"Model     : {os.path.basename(export_result.output)}")
        print(f"Input     : {export_result.input_image_shape} (rows, cols)")
        print(f"Latency   : {benchmark_result.latency}")
        print(f"Throughput: {benchmark_result.throughput}")
        print(f"Extra     : {benchmark_result.extra}")
        if not benchmark_result.success:
            print(f"Error     : {benchmark_result}")
        assert benchmark_result.success

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
