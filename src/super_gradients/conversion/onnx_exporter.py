import typing

import torch
from super_gradients.common.registry.registry import register_exporter
from super_gradients.conversion import ExportParams, ExportTargetBackend
from super_gradients.conversion.abstract_exporter import AbstractExporter
from super_gradients.module_interfaces import ExportableObjectDetectionModel, ExportablePoseEstimationModel, ExportableSegmentationModel
from super_gradients.training.utils.export_utils import infer_image_shape_from_model
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


@register_exporter()
class ONNXRuntimeExporter(AbstractExporter):
    def __init__(self, output_path: str):
        """
        :param output_path: Output path for the exported model. Currently only supports ONNX format.
        """
        self.output_path = output_path

    def export(self, model, export_params: ExportParams):
        return self._internal_export(model, self.output_path, export_params, quantized_model=None)

    def export_quantized(self, original_model, quantized_result, export_params: ExportParams):
        return self._internal_export(original_model, self.output_path, export_params, quantized_model=quantized_result.quantized_model)

    def _internal_export(self, model, output_onnx_path, export_params: ExportParams, quantized_model) -> str:
        from super_gradients.conversion.onnx.export_to_onnx import export_to_onnx

        input_image_shape = export_params.input_image_shape
        if input_image_shape is None:
            input_image_shape = infer_image_shape_from_model(model)

        # if input_image_shape is None:
        #     input_image_shape = input_shape_from_dataloader[2:]
        #
        # input_channels = infer_image_input_channels(model)
        # if input_channels is not None and input_channels != input_shape_from_dataloader[1]:
        #     logger.warning("Inferred input channels does not match with the number of channels from the dataloader")

        export_result = None

        # A signatures of these two protocols are the same, so we can use the same method and set of parameters for both
        if isinstance(model, (ExportableObjectDetectionModel, ExportablePoseEstimationModel)):
            model = typing.cast(ExportableObjectDetectionModel, model)
            export_result = model.export(
                output=output_onnx_path,
                engine=ExportTargetBackend.ONNXRUNTIME,
                quantized_model=quantized_model,
                input_image_shape=input_image_shape,
                preprocessing=export_params.preprocessing,
                postprocessing=export_params.postprocessing,
                confidence_threshold=export_params.confidence_threshold,
                nms_threshold=export_params.detection_nms_iou_threshold,
                onnx_simplify=export_params.onnx_simplify,
                onnx_export_kwargs=export_params.onnx_export_kwargs,
                num_pre_nms_predictions=export_params.detection_num_pre_nms_predictions,
                max_predictions_per_image=export_params.detection_max_predictions_per_image,
                output_predictions_format=export_params.detection_predictions_format,
            )
        elif isinstance(model, ExportableSegmentationModel):
            model: ExportableSegmentationModel = typing.cast(ExportableSegmentationModel, model)
            export_result = model.export(
                output=export_params.output_onnx_path,
                engine=ExportTargetBackend.ONNXRUNTIME,
                quantized_model=quantized_model,
                input_image_shape=input_image_shape,
                preprocessing=export_params.preprocessing,
                postprocessing=export_params.postprocessing,
                confidence_threshold=export_params.confidence_threshold,
                onnx_simplify=export_params.onnx_simplify,
                onnx_export_kwargs=export_params.onnx_export_kwargs,
            )
        else:
            device = "cpu"
            input_shape_with_explicit_batch = tuple([export_params.batch_size] + list(input_image_shape[1:]))
            onnx_input = torch.randn(input_shape_with_explicit_batch).to(device=device)
            onnx_export_kwargs = export_params.onnx_export_kwargs or {}
            model_to_export = quantized_model or model
            export_to_onnx(
                model=model_to_export.to(device),
                model_input=onnx_input,
                onnx_filename=export_params.output_onnx_path,
                input_names=["input"],
                onnx_opset=onnx_export_kwargs.get("opset_version", None),
                do_constant_folding=onnx_export_kwargs.get("do_constant_folding", True),
                dynamic_axes=onnx_export_kwargs.get("dynamic_axes", None),
                keep_initializers_as_inputs=onnx_export_kwargs.get("keep_initializers_as_inputs", False),
                verbose=onnx_export_kwargs.get("verbose", False),
            )

        return export_result
