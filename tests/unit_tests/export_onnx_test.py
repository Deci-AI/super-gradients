import logging
import tempfile
import unittest

import cv2
import numpy as np
import onnxruntime

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.utils.media.image import load_image
from torchvision.transforms import Compose, Normalize, Resize

from super_gradients.training.dataloaders import coco2017_val
from super_gradients.training.transforms import Standardize
import os

from super_gradients.training.utils.export_utils import infer_image_shape_from_model


class TestModelsONNXExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)

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

    def test_export_ppyolo_e_onnx_nms(self):
        # with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "."
        ppyolo_e = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

        onnx_file = os.path.join(tmpdirname, "ppyoloe_s_torchvision_nms.onnx")
        ppyolo_e.export(
            onnx_file,
            batch_size=1,
            image_shape=(640, 640),
            preprocessing=False,
            postprocessing=True,
            quantize=False,
        )

        image = load_image("../data/tinycoco/images/val2017/000000444010.jpg")
        image = cv2.resize(image, (640, 640))
        image = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2)) / 255.0

        session = onnxruntime.InferenceSession(onnx_file)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: image.astype(np.float32)})
        for r in result:
            print(r.shape, r.dtype, r)

    def test_export_ppyolo_e_onnx_nms_and_benchmark(self):
        tmpdirname = "."
        ppyolo_e = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

        onnx_file = os.path.join(tmpdirname, "ppyoloe_s_torchvision_nms.onnx")
        ppyolo_e.export(
            onnx_file,
            batch_size=1,
            image_shape=(640, 640),
            preprocessing=False,
            postprocessing=True,
            quantize=False,
        )

        image = load_image("../data/tinycoco/images/val2017/000000444010.jpg")
        image = cv2.resize(image, (640, 640))
        image = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2)) / 255.0

        session = onnxruntime.InferenceSession(onnx_file)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: image.astype(np.float32)})
        for r in result:
            print(r.shape, r.dtype, r)

        self._benchmark_onnx(onnx_file)

    def test_export_ppyolo_e_trt_nms_and_benchmark(self):
        # with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = "."
        ppyolo_e = models.get(Models.PP_YOLOE_S, pretrained_weights="coco")

        onnx_file = os.path.join(tmpdirname, "ppyoloe_s_trt_nms.onnx")
        ppyolo_e.export(
            onnx_file,
            engine="tensorrt",
            batch_size=1,
            image_shape=(640, 640),
            preprocessing=False,
            postprocessing=True,
            quantize=False,
        )

        self._benchmark_onnx(onnx_file)

    def test_export_ppyolo_e_quantized(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ppyolo_e = models.get(Models.PP_YOLOE_S, pretrained_weights="coco").cuda()
            print(infer_image_shape_from_model(ppyolo_e))

            onnx_file = os.path.join(tmpdirname, "ppyoloe_s_quantized.onnx")
            ppyolo_e.export(
                onnx_file,
                image_shape=(640, 640),
                preprocessing=False,
                postprocessing=True,
                quantize=True,
            )

            session = onnxruntime.InferenceSession(onnx_file)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: np.random.rand(1, 3, 640, 640).astype(np.float32)})
            print(result, result[0].shape)

            self._benchmark_onnx(onnx_file)

    def test_export_ppyolo_e_quantized_with_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ppyolo_e = models.get(Models.PP_YOLOE_S, pretrained_weights="coco").cuda()
            print(infer_image_shape_from_model(ppyolo_e))

            calibration_loader = coco2017_val(dataset_params=dict(data_dir="e:/coco2017"), dataloader_params=dict(num_workers=0))

            onnx_file = os.path.join(tmpdirname, "ppyoloe_s_quantized.onnx")
            ppyolo_e.export(
                onnx_file,
                image_shape=(640, 640),
                preprocessing=False,
                postprocessing=True,
                quantize=True,
                calibration_loader=calibration_loader,
            )

            session = onnxruntime.InferenceSession(onnx_file)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: np.random.rand(1, 3, 640, 640).astype(np.float32)})
            print(result, result[0].shape)

    def _benchmark_onnx(self, onnx_file):
        from decibenchmark.api.client_manager import ClientManager
        from decibenchmark.api.hardware.jetson.jetson_device_filter import JetsonDeviceFilter
        from decibenchmark.common.hardware.jetson.jetson_model import JetsonModel
        from decibenchmark.common.execmethod.trt_exec_params import TrtExecParams

        # Create client manager
        client_manager = ClientManager.create()

        # Get jetson client
        client = client_manager.jetson

        job = client.benchmark.trt_exec(
            JetsonDeviceFilter(jetson_model=JetsonModel.AGX_ORIN),
            TrtExecParams(extra_cmd_params=["--fp16", "--avgRuns=100", "--duration=15"]),
        ).dispatch(onnx_file)

        result = job.wait_for_result()

        # Get the latency and throughput
        print(onnx_file)
        print(f"Latency: {result.latency}")
        print(f"Throughput: {result.throughput}")


if __name__ == "__main__":
    unittest.main()
