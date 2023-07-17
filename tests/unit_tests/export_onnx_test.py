import logging
import tempfile
import unittest

import numpy as np
import onnxruntime

from super_gradients.common.object_names import Models
from super_gradients.training import models
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

    def test_export_ppyolo_e(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ppyolo_e = models.get(Models.PP_YOLOE_S, pretrained_weights="coco").cuda()
            # print(infer_image_shape_from_model(ppyolo_e))

            onnx_file = os.path.join(tmpdirname, "ppyoloe_s_trt_nms.onnx")
            ppyolo_e.export(
                onnx_file,
                engine="tensorrt",
                image_shape=(640, 640),
                preprocessing=False,
                postprocessing=True,
                quantize=False,
            )

            session = onnxruntime.InferenceSession(onnx_file)
            inputs = [o.name for o in session.get_inputs()]
            outputs = [o.name for o in session.get_outputs()]
            result = session.run(outputs, {inputs[0]: np.random.rand(1, 3, 640, 640).astype(np.float32)})
            print(result, result[0].shape)

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


if __name__ == "__main__":
    unittest.main()
