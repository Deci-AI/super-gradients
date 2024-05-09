import logging
import os
import tempfile
import unittest

import numpy as np
import onnxruntime
import torch
from super_gradients.common.object_names import Models
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_install
from super_gradients.import_utils import import_pytorch_quantization_or_install
from super_gradients.module_interfaces import ExportableOpticalFlowModel, OpticalFlowModelExportResult
from super_gradients.training import models


gs = import_onnx_graphsurgeon_or_install()
import_pytorch_quantization_or_install()


class TestOpticalFlowModelExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)

        self.models_to_test = [
            Models.RAFT_S,
            Models.RAFT_L,
        ]

    def test_export_to_onnxruntime_and_run(self):
        """
        Test export to ONNX
        """

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in self.models_to_test:
                with self.subTest(model_type=model_type):
                    model_name = str(model_type).lower().replace(".", "_")
                    out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime.onnx")

                    model_arch: ExportableOpticalFlowModel = models.get(model_name, num_classes=1)
                    export_result = model_arch.export(
                        out_path,
                        input_image_shape=(640, 640),  # Force .export() to infer image shape from the model itself
                        input_image_channels=3,
                        input_image_dtype=torch.float32,
                        onnx_export_kwargs={"opset_version": 16},
                    )

                    [flow_prediction] = self._run_inference_with_onnx(export_result)
                    self.assertTrue(flow_prediction.shape[0] == 1)
                    self.assertTrue(flow_prediction.shape[1] == 2)
                    self.assertTrue(flow_prediction.shape[2] == 640)
                    self.assertTrue(flow_prediction.shape[3] == 640)

    @staticmethod
    def _run_inference_with_onnx(export_result: OpticalFlowModelExportResult):
        input = np.zeros((1, 2, 3, 640, 640)).astype(np.float32)

        session = onnxruntime.InferenceSession(export_result.output)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: input})

        return result


if __name__ == "__main__":
    unittest.main()
