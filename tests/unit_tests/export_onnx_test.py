import logging
import os
import tempfile
import unittest

import cv2
import numpy as np
import onnx
import onnxruntime
import onnx_graphsurgeon as gs
import torch
from torchvision.transforms import Compose, Normalize, Resize

from super_gradients.common.object_names import Models
from super_gradients.conversion.onnx.nms import (
    ConvertFlatTensorToTRTFormat,
    PickNMSPredictionsAndReturnAsFlatResult,
    PickNMSPredictionsAndReturnAsBatchedResult,
)
from super_gradients.conversion.tensorrt.nms import ConvertTRTFormatToFlatTensor
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val  # noqa
from super_gradients.training.transforms import Standardize
from super_gradients.training.utils.media.image import load_image


class TestModelsONNXExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)

        try:
            from decibenchmark.api.client_manager import ClientManager  # noqa

            self.decibenchmark_available = True
        except ImportError:
            self.decibenchmark_available = False

    def test_models_onnx_export_with_deprecated_input_shape(self):
        pretrained_model = models.get(Models.RESNET18, num_classes=1000, pretrained_weights="imagenet")
        preprocess = Compose([Resize(224), Standardize(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "resnet18.onnx")
            models.convert_to_onnx(model=pretrained_model, out_path=out_path, input_shape=(3, 256, 256), pre_process=preprocess)
            self.assertTrue(os.path.exists(out_path))

    def test_models_onnx_export(self):
        pretrained_model = models.get(Models.RESNET18, num_classes=1000, pretrained_weights="imagenet")
        preprocess = Compose([Resize(224), Standardize(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "resnet18.onnx")
            models.convert_to_onnx(
                model=pretrained_model, out_path=out_path, pre_process=preprocess, prep_model_for_conversion_kwargs=dict(input_size=(1, 3, 640, 640))
            )
            self.assertTrue(os.path.exists(out_path))

    def _export_and_benchmark(self, onnx_filename: str, run_benchmark: bool, run_inference_with_onnxruntime: bool, export_kwargs=None, benchmark_kwargs=None):
        if export_kwargs is None:
            export_kwargs = {}
        if benchmark_kwargs is None:
            benchmark_kwargs = {}

        ppyolo_e = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

        ppyolo_e.export(
            onnx_filename,
            **export_kwargs,
        )

        if run_inference_with_onnxruntime:
            image = load_image("../data/tinycoco/images/val2017/000000444010.jpg")
            image = cv2.resize(image, (640, 640))
            image = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2)) / 255.0

            session = onnxruntime.InferenceSession(onnx_filename)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: image.astype(np.float32)})
            for r in result:
                print(r.shape, r.dtype, r)

        if run_benchmark:
            self._benchmark_onnx(onnx_filename, **benchmark_kwargs)

    # def test_export_ppyolo_e_all_export_variants(self):
    #     for output_predictions_format in ["batched"]:
    #         for engine in {"tensorrt"}:
    #             for quantize in {False}:
    #                 precision = "quantized" if quantize else "full_precision"
    #                 self._export_and_benchmark(
    #                     onnx_filename=f"ppyoloe_s_{engine}_engine_{output_predictions_format}_format_{precision}.onnx",
    #                     run_benchmark=False,
    #                     run_inference_with_onnxruntime=engine != "tensorrt",
    #                     export_kwargs=dict(
    #                         batch_size=1,
    #                         image_shape=(640, 640),
    #                         preprocessing=False,
    #                         postprocessing=True,
    #                         quantize=quantize,
    #                         engine=engine,
    #                         output_predictions_format=output_predictions_format,
    #                     ),
    #                     benchmark_kwargs=dict(precision="--int8" if quantize else "--fp16"),
    #                 )

    def test_export_ppyoloe_onnxruntime_engine_batched_output(self):
        quantize = False
        engine = "onnx"
        output_predictions_format = "batched"
        precision = "quantized" if quantize else "full_precision"

        self._export_and_benchmark(
            onnx_filename=f"ppyoloe_s_{engine}_engine_{output_predictions_format}_format_{precision}.onnx",
            run_benchmark=self.decibenchmark_available,
            run_inference_with_onnxruntime=True,
            export_kwargs=dict(
                batch_size=1,
                image_shape=(640, 640),
                preprocessing=False,
                postprocessing=True,
                quantize=quantize,
                engine=engine,
                output_predictions_format=output_predictions_format,
            ),
            benchmark_kwargs=dict(precision="--int8" if quantize else "--fp16"),
        )

    def test_export_ppyoloe_onnxruntime_engine_flat_output(self):
        quantize = False
        engine = "onnx"
        output_predictions_format = "flat"
        precision = "quantized" if quantize else "full_precision"

        self._export_and_benchmark(
            onnx_filename=f"ppyoloe_s_{engine}_engine_{output_predictions_format}_format_{precision}.onnx",
            run_benchmark=self.decibenchmark_available,
            run_inference_with_onnxruntime=True,
            export_kwargs=dict(
                batch_size=1,
                image_shape=(640, 640),
                preprocessing=False,
                postprocessing=True,
                quantize=quantize,
                engine=engine,
                output_predictions_format=output_predictions_format,
            ),
            benchmark_kwargs=dict(precision="--int8" if quantize else "--fp16"),
        )

    def test_export_ppyoloe_trt_engine_batched_output(self):
        quantize = False
        engine = "tensorrt"
        output_predictions_format = "batched"
        precision = "quantized" if quantize else "full_precision"

        self._export_and_benchmark(
            onnx_filename=f"ppyoloe_s_{engine}_engine_{output_predictions_format}_format_{precision}.onnx",
            run_benchmark=self.decibenchmark_available,
            run_inference_with_onnxruntime=False,
            export_kwargs=dict(
                batch_size=1,
                image_shape=(640, 640),
                preprocessing=False,
                postprocessing=True,
                quantize=quantize,
                engine=engine,
                output_predictions_format=output_predictions_format,
            ),
            benchmark_kwargs=dict(precision="--int8" if quantize else "--fp16"),
        )

    def test_export_ppyoloe_trt_engine_flat_output(self):
        quantize = False
        engine = "tensorrt"
        output_predictions_format = "flat"
        precision = "quantized" if quantize else "full_precision"

        self._export_and_benchmark(
            onnx_filename=f"ppyoloe_s_{engine}_engine_{output_predictions_format}_format_{precision}.onnx",
            run_benchmark=self.decibenchmark_available,
            run_inference_with_onnxruntime=False,
            export_kwargs=dict(
                batch_size=1,
                image_shape=(640, 640),
                preprocessing=False,
                postprocessing=True,
                quantize=quantize,
                engine=engine,
                output_predictions_format=output_predictions_format,
            ),
            benchmark_kwargs=dict(precision="--int8" if quantize else "--fp16"),
        )

    # def test_export_ppyolo_e_quantized_with_calibration(self):
    #     quantize = True
    #     engine = "onnx"
    #     output_predictions_format = "flat"
    #     precision = "quantized" if quantize else "full_precision"
    #     calibration_loader = coco2017_val(dataset_params=dict(data_dir="e:/coco2017"), dataloader_params=dict(num_workers=0))
    #
    #     self._export_and_benchmark(
    #         onnx_filename=f"ppyoloe_s_{engine}_engine_{output_predictions_format}_format_{precision}_calibrated.onnx",
    #         run_benchmark=True,
    #         run_inference_with_onnxruntime=engine != "tensorrt",
    #         export_kwargs=dict(
    #             batch_size=1,
    #             image_shape=(640, 640),
    #             preprocessing=False,
    #             postprocessing=True,
    #             quantize=quantize,
    #             calibration_loader=calibration_loader,
    #             output_predictions_format=output_predictions_format,
    #         ),
    #         benchmark_kwargs=dict(precision="--int8" if quantize else "--fp16"),
    #     )

    def test_onnx_nms_flat_result(self):
        onnx_file = "PickNMSPredictionsAndReturnAsFlatResult.onnx"
        graph = PickNMSPredictionsAndReturnAsFlatResult.as_graph()
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

    def test_flat_tensor_to_trt_format(self):
        predictions = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [0, 11, 12, 13, 14, 15, 16], [1, 21, 22, 23, 24, 25, 26], [2, 31, 32, 33, 34, 35, 36], [3, 31, 32, 33, 34, 35, 36]]
        ).float()

        flat_to_batched_predictions = ConvertFlatTensorToTRTFormat(batch_size=4, max_predictions_per_image=20)
        batched_to_flat_predictions = ConvertTRTFormatToFlatTensor(batch_size=4)

        num_predictions, pred_boxes, pred_scores, pred_classes = flat_to_batched_predictions(predictions)
        np.testing.assert_allclose(predictions, batched_to_flat_predictions(num_predictions, pred_boxes, pred_scores, pred_classes))

        num_predictions, pred_boxes, pred_scores, pred_classes = flat_to_batched_predictions(torch.zeros((0, 7), dtype=torch.float32))
        np.testing.assert_allclose(
            torch.zeros((0, 7), dtype=torch.float32), batched_to_flat_predictions(num_predictions, pred_boxes, pred_scores, pred_classes)
        )

        torch.onnx.export(
            ConvertFlatTensorToTRTFormat(batch_size=4, max_predictions_per_image=20),
            args=predictions,
            f="ConvertFlatTensorToTRTFormat.onnx",
            input_names=["flat_predictions"],
            output_names=["num_predictions", "pred_boxes", "pred_scores", "pred_classes"],
            dynamic_axes={"flat_predictions": {0: "num_predictions"}},
        )

        onnx.checker.check_model(onnx.load("ConvertFlatTensorToTRTFormat.onnx"))

        session = onnxruntime.InferenceSession("ConvertFlatTensorToTRTFormat.onnx")

        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]

        result = session.run(outputs, {inputs[0]: predictions.numpy()})
        for r in result:
            print(r.shape, r.dtype, r)

        result = session.run(outputs, {inputs[0]: np.zeros((1, 7), dtype=np.float32)})
        for r in result:
            print(r.shape, r.dtype, r)

        result = session.run(outputs, {inputs[0]: np.zeros((0, 7), dtype=np.float32)})
        for r in result:
            print(r.shape, r.dtype, r)

    def test_trt_format_to_flat_format(self):
        predictions = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [0, 11, 12, 13, 14, 15, 16], [0, 21, 22, 23, 24, 25, 26], [2, 31, 32, 33, 34, 35, 36], [3, 31, 32, 33, 34, 35, 36]]
        ).float()

        flat_to_batched_predictions = ConvertFlatTensorToTRTFormat(batch_size=4, max_predictions_per_image=20)
        batched_to_flat_predictions = ConvertTRTFormatToFlatTensor(batch_size=4)

        num_predictions, pred_boxes, pred_scores, pred_classes = flat_to_batched_predictions(torch.zeros((0, 7), dtype=torch.float32))
        assert num_predictions.eq(0).all()
        assert pred_boxes.eq(0).all()
        assert pred_scores.eq(0).all()
        assert pred_classes.eq(0).all()
        preds = batched_to_flat_predictions(num_predictions, pred_boxes, pred_scores, pred_classes)
        assert preds.shape == (0, 7)

        num_predictions, pred_boxes, pred_scores, pred_classes = flat_to_batched_predictions(predictions)
        assert num_predictions[0].item() == 3
        assert num_predictions[1].item() == 0
        assert num_predictions[2].item() == 1
        assert num_predictions[3].item() == 1

        batch_size = 4
        max_predictions_per_image = 100
        num_predictions = torch.zeros((batch_size, 1), dtype=torch.int64)
        pred_boxes = torch.zeros((batch_size, max_predictions_per_image, 4), dtype=torch.float32)
        pred_scores = torch.zeros((batch_size, max_predictions_per_image), dtype=torch.float32)
        pred_classes = torch.zeros((batch_size, max_predictions_per_image), dtype=torch.int64)

        torch.onnx.export(
            ConvertTRTFormatToFlatTensor(batch_size=4),
            args=(num_predictions, pred_boxes, pred_scores, pred_classes),
            f="ConvertTRTFormatToFlatTensor.onnx",
            input_names=["num_predictions", "pred_boxes", "pred_scores", "pred_classes"],
            output_names=["flat_predictions"],
            dynamic_axes={"flat_predictions": {0: "num_predictions"}},
        )

        onnx.checker.check_model(onnx.load("ConvertTRTFormatToFlatTensor.onnx"))

        session = onnxruntime.InferenceSession("ConvertTRTFormatToFlatTensor.onnx")

        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]

        result = session.run(
            outputs, {inputs[0]: num_predictions.numpy(), inputs[1]: pred_boxes.numpy(), inputs[2]: pred_scores.numpy(), inputs[3]: pred_classes.numpy()}
        )
        for r in result:
            print(r.shape, r.dtype, r)

    def _benchmark_onnx(self, onnx_file, precision="--fp16"):
        from decibenchmark.api.client_manager import ClientManager
        from decibenchmark.api.hardware.jetson.jetson_device_filter import JetsonDeviceFilter

        # from decibenchmark.api.hardware.nvidia_gpu.nvidia_gpu_device_filter import NvidiaGpuDeviceFilter
        from decibenchmark.common.hardware.jetson.jetson_model import JetsonModel
        from decibenchmark.common.execmethod.trt_exec_params import TrtExecParams

        # from decibenchmark.common.hardware.nvidia_gpu.nvidia_gpu_model import NvidiaGpuModel

        # Create client manager
        client_manager = ClientManager.create()

        # Get jetson client
        client = client_manager.jetson

        job = client.benchmark.trt_exec(
            JetsonDeviceFilter(jetson_model=JetsonModel.XAVIER_NX),
            # NvidiaGpuDeviceFilter(nvidia_gpu_model=NvidiaGpuModel.TESLA_T4),
            TrtExecParams(extra_cmd_params=[precision, "--avgRuns=100", "--duration=15"]),
        ).dispatch(onnx_file)

        result = job.wait_for_result(timeout=-1)
        # print(result)
        print(onnx_file)
        # Get the latency and throughput
        print(f"Latency: {result.latency}")
        print(f"Throughput: {result.throughput}")


if __name__ == "__main__":
    unittest.main()
