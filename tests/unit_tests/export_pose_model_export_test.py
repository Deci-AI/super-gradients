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
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_fail_with_instructions
from super_gradients.conversion.onnx.pose_nms import PoseNMSAndReturnAsBatchedResult, PoseNMSAndReturnAsFlatResult
from super_gradients.module_interfaces import ExportablePoseEstimationModel, PoseEstimationModelExportResult
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val  # noqa
from super_gradients.training.pretrained_models import MODEL_URLS
from super_gradients.training.processing.processing import default_yolo_nas_pose_coco_processing_params
from super_gradients.training.utils.export_utils import infer_image_shape_from_model, infer_image_input_channels
from super_gradients.training.utils.media.image import load_image
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization

gs = import_onnx_graphsurgeon_or_fail_with_instructions()


class TestPoseEstimationModelExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        this_dir = os.path.dirname(__file__)
        self.test_image_path = os.path.join(this_dir, "../data/tinycoco/images/val2017/000000444010.jpg")
        self.default_pretrained_weights = "coco_pose"
        self.default_model = Models.YOLO_NAS_POSE_S
        MODEL_URLS[Models.YOLO_NAS_POSE_S + "_coco_pose"] = "file:///G:/super-gradients/checkpoints/coco2017_yolo_nas_pose_s_mosaic_v2_average_model.pth"
        params = default_yolo_nas_pose_coco_processing_params()
        self.edge_links = params["edge_links"]
        self.edge_colors = params["edge_colors"]
        self.keypoint_colors = params["keypoint_colors"]

    def test_export_model_on_small_size(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_POSE_S,
            ]:
                out_path = os.path.join(tmpdirname, model_type + ".onnx")
                model: ExportablePoseEstimationModel = models.get(model_type, pretrained_weights=self.default_pretrained_weights)
                result = model.export(
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
            out_path = os.path.join(tmpdirname, "model.onnx")

            model: ExportablePoseEstimationModel = models.get(self.default_model, pretrained_weights=self.default_pretrained_weights)
            result = model.export(out_path)
            assert result.input_image_dtype == torch.uint8
            assert result.input_image_shape == (640, 640)
            assert result.input_image_channels == 3

    def test_models_produce_half(self):
        if not torch.cuda.is_available():
            self.skipTest("This test was skipped because target machine has not CUDA devices")

        input = torch.randn(1, 3, 640, 640).half().cuda()

        model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17, pretrained_weights=None)
        model = nn.Sequential(model, model.get_decoding_module(100)).cuda().eval().half()
        output = model(input)
        assert output[0].dtype == torch.float16
        assert output[1].dtype == torch.float16

    def test_infer_input_image_shape_from_model(self):
        assert infer_image_shape_from_model(models.get(Models.YOLO_NAS_POSE_S, num_classes=17, pretrained_weights=None)) is None
        assert infer_image_shape_from_model(models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)) == (640, 640)

    def test_infer_input_image_num_channels_from_model(self):
        assert infer_image_input_channels(models.get(Models.YOLO_NAS_POSE_S, num_classes=17, pretrained_weights=None)) == 3
        assert infer_image_input_channels(models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)) == 3

    def test_export_to_onnxruntime_flat(self):
        """
        Test export to ONNX with flat predictions
        """
        output_predictions_format = DetectionOutputFormatMode.FLAT_FORMAT
        confidence_threshold = 0.7
        nms_threshold = 0.6

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_POSE_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_flat.onnx")

                model_arch: ExportablePoseEstimationModel = models.get(model_name, pretrained_weights=self.default_pretrained_weights)
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
                Models.YOLO_NAS_POSE_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_batch.onnx")

                model_arch: ExportablePoseEstimationModel = models.get(model_name, pretrained_weights=self.default_pretrained_weights)
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
                Models.YOLO_NAS_POSE_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_flat.onnx")

                model_arch: ExportablePoseEstimationModel = models.get(model_name, pretrained_weights=self.default_pretrained_weights)
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
                Models.YOLO_NAS_POSE_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_batch.onnx")

                model_arch: ExportablePoseEstimationModel = models.get(model_name, pretrained_weights=self.default_pretrained_weights)
                export_result = model_arch.export(
                    out_path,
                    input_image_shape=None,  # Force .export() to infer image shape from the model itself
                    engine=ExportTargetBackend.TENSORRT,
                    output_predictions_format=output_predictions_format,
                    nms_threshold=nms_threshold,
                    confidence_threshold=confidence_threshold,
                )
                assert export_result is not None

    def test_export_to_tensorrt_batch_format_YOLO_NAS_POSE_S(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.25
        nms_threshold = 0.6
        model_type = Models.YOLO_NAS_POSE_S
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_name = str(model_type).lower().replace(".", "_")
            out_path = os.path.join(tmpdirname, f"{model_name}_tensorrt_batch.onnx")

            model_arch: ExportablePoseEstimationModel = models.get(model_name, pretrained_weights=self.default_pretrained_weights)
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

            model: ExportablePoseEstimationModel = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)
            export_result = model.export(out_path, engine=ExportTargetBackend.ONNXRUNTIME, input_image_shape=(320, 320), output_predictions_format="flat")
            [flat_predictions] = self._run_inference_with_onnx(export_result)

            bbox_dims = 4
            pose_score_dims = 1
            pose_coords_dims = 17 * 3
            assert flat_predictions.shape[1] == bbox_dims + pose_score_dims + pose_coords_dims

    def test_export_with_fp16_quantization(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            self.skipTest("No CUDA or MPS device available")
            return

        max_predictions_per_image = 300
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "model_with_fp16_quantization.onnx")

            model: ExportablePoseEstimationModel = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)
            export_result = model.export(
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
            out_path = os.path.join(tmpdirname, "model_s_with_fp16_quantization.onnx")

            model: ExportablePoseEstimationModel = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)
            export_result = model.export(
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
            out_path = os.path.join(tmpdirname, "model_s_with_int8_quantization.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8, num_workers=0)

            ppyolo_e: ExportablePoseEstimationModel = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)
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
            out_path = os.path.join(tmpdirname, "model_quantized_with_calibration.onnx")

            dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
            dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)

            ppyolo_e: ExportablePoseEstimationModel = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)
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

            ppyolo_e: ExportablePoseEstimationModel = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)
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

    def _run_inference_with_onnx(self, export_result: PoseEstimationModelExportResult):
        # onnx_filename = out_path, input_shape = export_result.image_shape, output_predictions_format = output_predictions_format

        image = self._get_image_as_bchw(export_result.input_image_shape)
        image_8u = self._get_image(export_result.input_image_shape)

        session = onnxruntime.InferenceSession(export_result.output)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: image})

        if export_result.output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
            flat_predictions = result[0]  # [N, (batch_index, x1, y1, x2, y2, score, class]
            assert flat_predictions.shape[1] == 1 + 4 + 1 + 17 * 3

            boxes = flat_predictions[:, 1:5]
            scores = flat_predictions[:, 5]
            poses = flat_predictions[:, 6:].reshape(-1, 17, 3)

            image_8u = PoseVisualization.draw_poses(
                image_8u,
                poses=poses,
                boxes=boxes,
                scores=scores,
                show_keypoint_confidence=True,
                edge_links=self.edge_links,
                edge_colors=self.edge_colors,
                keypoint_colors=self.keypoint_colors,
            )

        else:
            # Hard-coded unpacking for batch size 1
            [num_predictions], [pred_boxes], [pred_scores], [pred_joints] = result

            image_8u = PoseVisualization.draw_poses(
                image_8u,
                poses=pred_joints[0 : num_predictions[0]],
                boxes=pred_boxes[0 : num_predictions[0]],
                scores=pred_scores[0 : num_predictions[0]],
                show_keypoint_confidence=True,
                edge_links=self.edge_links,
                edge_colors=self.edge_colors,
                keypoint_colors=self.keypoint_colors,
            )

        plt.figure(figsize=(10, 10))
        plt.imshow(image_8u)
        plt.title(os.path.basename(export_result.output))
        plt.tight_layout()
        plt.show()

        return result

    def test_export_already_quantized_model(self):
        model = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights=self.default_pretrained_weights)
        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights="max",
            default_quant_modules_calibrator_inputs="histogram",
            default_per_channel_quant_weights=True,
            default_learn_amax=False,
            verbose=True,
        )
        q_util.quantize_module(model)

        with tempfile.TemporaryDirectory() as tmpdirname:
            output_model1 = os.path.join(tmpdirname, "YOLO_NAS_POSE_S_quantized_explicit_int8.onnx")
            output_model2 = os.path.join(tmpdirname, "YOLO_NAS_POSE_S_quantized.onnx")

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
                    "YOLO_NAS_POSE_S_quantized.onnx",
                    quantization_mode=ExportQuantizationMode.FP16,
                )

            # Assert two files are the same
            # with open(output_model1, "rb") as f1, open(output_model2, "rb") as f2:
            #     assert hashlib.md5(f1.read()) == hashlib.md5(f2.read())

    def test_onnx_nms_flat_result(self):
        max_predictions = 100
        batch_size = 7
        num_joints = 17

        if torch.cuda.is_available():
            available_devices = ["cpu", "cuda"]
            available_dtypes = [torch.float16, torch.float32]
        else:
            available_devices = ["cpu"]
            available_dtypes = [torch.float32]

        for device in available_devices:
            for dtype in available_dtypes:

                # Run a few tests to ensure ONNX model produces the same results as the PyTorch model
                # And also can handle dynamic shapes input
                pred_boxes = torch.randn((batch_size, max_predictions, 4), dtype=dtype)
                pred_scores = torch.randn((batch_size, max_predictions, 1), dtype=dtype)
                pred_joints = torch.randn((batch_size, max_predictions, num_joints, 3), dtype=dtype)
                selected_indexes = torch.tensor([[6, 0, 4], [1, 0, 3], [2, 0, 2], [2, 0, 1]], dtype=torch.int64)

                torch_module = PoseNMSAndReturnAsFlatResult(
                    batch_size=batch_size, num_pre_nms_predictions=max_predictions, max_predictions_per_image=max_predictions
                )
                torch_result = torch_module(pred_boxes, pred_scores, pred_joints, selected_indexes)

                with tempfile.TemporaryDirectory() as temp_dir:
                    onnx_file = os.path.join(temp_dir, "PoseNMSAndReturnAsFlatResult.onnx")
                    graph = PoseNMSAndReturnAsFlatResult.as_graph(
                        batch_size=batch_size, num_pre_nms_predictions=max_predictions, max_predictions_per_image=max_predictions, device=device, dtype=dtype
                    )

                    model = gs.export_onnx(graph)
                    onnx.checker.check_model(model)
                    onnx.save(model, onnx_file)

                    session = onnxruntime.InferenceSession(onnx_file)

                    inputs = [o.name for o in session.get_inputs()]
                    outputs = [o.name for o in session.get_outputs()]

                    [onnx_result] = session.run(
                        outputs,
                        {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: pred_joints.numpy(), inputs[3]: selected_indexes.numpy()},
                    )

                    np.testing.assert_allclose(torch_result.numpy(), onnx_result, rtol=1e-3, atol=1e-3)

    def test_onnx_nms_batch_result(self):
        max_predictions = 100
        batch_size = 7
        num_joints = 17

        if torch.cuda.is_available():
            available_devices = ["cpu", "cuda"]
            available_dtypes = [torch.float16, torch.float32]
        else:
            available_devices = ["cpu"]
            available_dtypes = [torch.float32]

        for device in available_devices:
            for dtype in available_dtypes:

                # Run a few tests to ensure ONNX model produces the same results as the PyTorch model
                # And also can handle dynamic shapes input
                pred_boxes = torch.randn((batch_size, max_predictions, 4), dtype=dtype)
                pred_scores = torch.randn((batch_size, max_predictions, 1), dtype=dtype)
                pred_joints = torch.randn((batch_size, max_predictions, num_joints, 3), dtype=dtype)
                selected_indexes = torch.tensor([[6, 0, 4], [1, 0, 3], [2, 0, 2], [2, 0, 1]], dtype=torch.int64)

                torch_module = PoseNMSAndReturnAsBatchedResult(
                    batch_size=batch_size, num_pre_nms_predictions=max_predictions, max_predictions_per_image=max_predictions
                )
                torch_result = torch_module(pred_boxes, pred_scores, pred_joints, selected_indexes)

                with tempfile.TemporaryDirectory() as temp_dir:
                    onnx_file = os.path.join(temp_dir, "PoseNMSAndReturnAsBatchedResult.onnx")
                    graph = PoseNMSAndReturnAsBatchedResult.as_graph(
                        batch_size=batch_size, num_pre_nms_predictions=max_predictions, max_predictions_per_image=max_predictions, device=device, dtype=dtype
                    )

                    model = gs.export_onnx(graph)
                    onnx.checker.check_model(model)
                    onnx.save(model, onnx_file)

                    session = onnxruntime.InferenceSession(onnx_file)

                    inputs = [o.name for o in session.get_inputs()]
                    outputs = [o.name for o in session.get_outputs()]

                    onnx_result = session.run(
                        outputs,
                        {inputs[0]: pred_boxes.numpy(), inputs[1]: pred_scores.numpy(), inputs[2]: pred_joints.numpy(), inputs[3]: selected_indexes.numpy()},
                    )

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
