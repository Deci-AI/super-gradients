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

    # def test_infer_input_image_shape_from_model(self):
    #     assert infer_image_shape_from_model(models.get(Models.RAFT_S, num_classes=1)) is None
    #     assert infer_image_shape_from_model(models.get(Models.RAFT_L, num_classes=1)) is None

    # def test_infer_input_image_num_channels_from_model(self):
    #     assert infer_image_input_channels(models.get(Models.RAFT_S, num_classes=1)) == 3
    #     assert infer_image_input_channels(models.get(Models.RAFT_L, num_classes=1)) == 3

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
                    )

                    [flow_prediction] = self._run_inference_with_onnx(export_result)
                    self.assertTrue(flow_prediction.shape[0] == 1)
                    self.assertTrue(flow_prediction.shape[1] == 2)
                    self.assertTrue(flow_prediction.shape[2] == 640)
                    self.assertTrue(flow_prediction.shape[3] == 640)

    # def test_export_int8_quantized_with_calibration(self):
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         for model_type in self.models_to_test:
    #             with self.subTest(model_type=model_type):
    #                 model_name = str(model_type).lower().replace(".", "_")
    #                 out_path = os.path.join(tmpdirname, f"{model_name}.onnx")
    #
    #                 dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
    #                 dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8)
    #
    #                 model_arch: ExportableOpticalFlowModel = models.get(model_name, num_classes=1)
    #                 export_result = model_arch.export(
    #                     out_path,
    #                     input_image_shape=(640, 640),  # Force .export() to infer image shape from the model itself
    #                     quantization_mode=ExportQuantizationMode.INT8,
    #                     calibration_loader=dummy_calibration_loader,
    #                 )
    #
    #                 [flow_prediction] = self._run_inference_with_onnx(export_result)
    #                 self.assertTrue(flow_prediction.shape[0] == 1)
    #                 self.assertTrue(flow_prediction.shape[1] == 2)
    #                 self.assertTrue(flow_prediction.shape[2] == 640)
    #                 self.assertTrue(flow_prediction.shape[3] == 640)

    def _run_inference_with_onnx(self, export_result: OpticalFlowModelExportResult):
        # onnx_filename = out_path, input_shape = export_result.image_shape, output_predictions_format = output_predictions_format

        input = np.zeros((1, 2, 3, 640, 640)).astype(np.float32)

        session = onnxruntime.InferenceSession(export_result.output)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: input})

        return result

    # def test_export_already_quantized_model(self):
    #     from super_gradients.training.utils.quantization import SelectiveQuantizer
    #
    #     for model_type in self.models_to_test:
    #         with self.subTest(model_type=model_type):
    #             model = models.get(model_type, num_classes=1)
    #             q_util = SelectiveQuantizer(
    #                 default_quant_modules_calibrator_weights="max",
    #                 default_quant_modules_calibrator_inputs="histogram",
    #                 default_per_channel_quant_weights=True,
    #                 default_learn_amax=False,
    #                 verbose=True,
    #             )
    #             q_util.quantize_module(model)
    #
    #             with tempfile.TemporaryDirectory() as tmpdirname:
    #                 output_model1 = os.path.join(tmpdirname, f"{model_type}_quantized_explicit_int8.onnx")
    #                 output_model2 = os.path.join(tmpdirname, f"{model_type}_quantized.onnx")
    #
    #                 # If model is already quantized to int8, the export should be successful but model should not be quantized again
    #                 model.export(
    #                     output_model1,
    #                     quantization_mode=ExportQuantizationMode.INT8,
    #                 )
    #
    #                 # If model is quantized but quantization mode is not specified, the export should be also successful
    #                 # but model should not be quantized again
    #                 model.export(
    #                     output_model2,
    #                     quantization_mode=None,
    #                 )
    #
    #                 # If model is already quantized to int8, we should not be able to export model to FP16
    #                 with self.assertRaises(RuntimeError):
    #                     model.export(
    #                         "raft_s_quantized.onnx",
    #                         quantization_mode=ExportQuantizationMode.FP16,
    #                     )


if __name__ == "__main__":
    unittest.main()
