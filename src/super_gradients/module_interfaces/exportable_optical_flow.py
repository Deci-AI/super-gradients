import copy
import dataclasses
import gc
from typing import Union, Optional, List, Tuple

import numpy as np
import onnx
import onnxsim
import torch
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion import ExportTargetBackend, ExportQuantizationMode
from super_gradients.conversion.conversion_utils import find_compatible_model_device_for_dtype
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_install
from super_gradients.import_utils import import_pytorch_quantization_or_install
from super_gradients.module_interfaces.supports_input_shape_check import SupportsInputShapeCheck
from super_gradients.training.utils.export_utils import (
    infer_image_shape_from_model,
    infer_image_input_channels,
)
from super_gradients.training.utils.utils import infer_model_device, check_model_contains_quantized_modules
from super_gradients.conversion.onnx.export_to_onnx import export_to_onnx
from torch import nn
from torch.utils.data import DataLoader

logger = get_logger(__name__)

__all__ = ["ExportableOpticalFlowModel", "OpticalFlowModelExportResult"]


@dataclasses.dataclass
class OpticalFlowModelExportResult:
    """
    A dataclass that holds the result of model export.
    """

    input_image_channels: int
    input_image_dtype: torch.dtype
    input_image_shape: Tuple[int, int]

    engine: ExportTargetBackend
    quantization_mode: Optional[ExportQuantizationMode]

    output: str

    usage_instructions: str = ""

    def __repr__(self):
        return self.usage_instructions


class ExportableOpticalFlowModel:
    """
    A mixin class that adds export functionality to the optical flow models.
    Classes that inherit from this mixin must implement the following methods:
    - get_decoding_module()
    - get_preprocessing_callback()
    Providing these methods are implemented correctly, the model can be exported to ONNX or TensorRT formats
    using model.export(...) method.
    """

    def export(
        self,
        output: str,
        quantization_mode: Optional[ExportQuantizationMode] = None,
        selective_quantizer: Optional["SelectiveQuantizer"] = None,  # noqa
        calibration_loader: Optional[DataLoader] = None,
        calibration_method: str = "percentile",
        calibration_batches: int = 16,
        calibration_percentile: float = 99.99,
        batch_size: int = 1,
        input_image_shape: Optional[Tuple[int, int]] = None,
        input_image_channels: Optional[int] = None,
        input_image_dtype: Optional[torch.dtype] = None,
        onnx_export_kwargs: Optional[dict] = None,
        onnx_simplify: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Export the model to one of supported formats. Format is inferred from the output file extension or can be
        explicitly specified via `format` argument.

        :param output: Output file name of the exported model.
        :param quantization_mode: (QuantizationMode) Sets the quantization mode for the exported model.
            If None, the model is exported as-is without any changes to mode weights.
            If QuantizationMode.FP16, the model is exported with weights converted to half precision.
            If QuantizationMode.INT8, the model is exported with weights quantized to INT8 (Using PTQ).
            For this mode you can use calibration_loader to specify a data loader for calibrating the model.
        :param selective_quantizer: (SelectiveQuantizer) An optional quantizer for selectively quantizing model weights.
        :param calibration_loader: (torch.utils.data.DataLoader) An optional data loader for calibrating a quantized model.
        :param calibration_method: (str) Calibration method for quantized models. See QuantizationCalibrator for details.
        :param calibration_batches: (int) Number of batches to use for calibration. Default is 16.
        :param calibration_percentile: (float) Percentile for percentile calibration method. Default is 99.99.
        :param batch_size: (int) Batch size for the exported model.
        :param input_image_shape: (tuple) Input image shape (height, width) for the exported model.
               If None, the function will infer the image shape from the model's preprocessing params.
        :param input_image_channels: (int) Number of input image channels for the exported model.
               If None, the function will infer the number of channels from the model itself
               (No implemented now, will use hard-coded value of 3 for now).
        :param input_image_dtype: (torch.dtype) Type of the input image for the exported model.
                If None, the function will infer the dtype from the model's preprocessing and other parameters.
                If preprocessing is True, dtype will default to torch.uint8.
                If preprocessing is False and requested quantization mode is FP16 a torch.float16 will be used,
                otherwise a default torch.float32 dtype will be used.
        :param device: (torch.device) Device to use for exporting the model. If not specified, the device is inferred from the model itself.
        :param onnx_export_kwargs: (dict) Optional keyword arguments for torch.onnx.export() function.
        :param onnx_simplify: (bool) If True, apply onnx-simplifier to the exported model.
        :return:
        """

        # Do imports here to avoid raising error of missing onnx_graphsurgeon package if it is not needed.
        import_onnx_graphsurgeon_or_install()
        if ExportQuantizationMode.INT8 == quantization_mode:
            import_pytorch_quantization_or_install()
        from super_gradients.conversion.conversion_utils import torch_dtype_to_numpy_dtype

        usage_instructions = []

        # Hard-coded for now
        # Will be made a parameter if we decide to support CoreML/OpenVino/TRT export in the future
        engine = ExportTargetBackend.ONNXRUNTIME

        if not isinstance(self, nn.Module):
            raise TypeError(f"Export is only supported for torch.nn.Module. Got type {type(self)}")

        device: torch.device = device or infer_model_device(self)
        if device is None:
            raise ValueError(
                "Device is not specified and cannot be inferred from the model. "
                "Please specify the device explicitly: model.export(..., device=torch.device(...))"
            )

        # The following is a trick to infer the exact device index in order to make sure the model using right device.
        # User may pass device="cuda", which is not explicitly specifying device index.
        # Using this trick, we can infer the correct device (cuda:3 for instance) and use it later for checking
        # whether model places all it's parameters on the right device.
        device = torch.zeros(1).to(device).device

        logger.debug(f"Using device: {device} for exporting model {self.__class__.__name__}")

        model: nn.Module = copy.deepcopy(self).eval()

        # Infer the input image shape from the model
        if input_image_shape is None:
            input_image_shape = infer_image_shape_from_model(model)
            logger.debug(f"Inferred input image shape: {input_image_shape} from model {model.__class__.__name__}")

        if input_image_shape is None:
            raise ValueError(
                "Image shape is not specified and cannot be inferred from the model. "
                "Please specify the image shape explicitly: model.export(..., input_image_shape=(height, width))"
            )

        try:
            rows, cols = input_image_shape
        except ValueError:
            raise ValueError(f"Image shape must be a tuple of two integers (height, width), got {input_image_shape} instead")

        # Infer the number of input channels from the model
        if input_image_channels is None:
            input_image_channels = infer_image_input_channels(model)
            logger.debug(f"Inferred input image channels: {input_image_channels} from model {model.__class__.__name__}")

        if input_image_channels is None:
            raise ValueError(
                "Number of input channels is not specified and cannot be inferred from the model. "
                "Please specify the number of input channels explicitly: model.export(..., input_image_channels=NUM_CHANNELS_YOUR_MODEL_TAKES)"
            )

        input_shape = (batch_size, 2, input_image_channels, rows, cols)

        if isinstance(model, SupportsInputShapeCheck):
            model.validate_input_shape(input_shape)

        prep_model_for_conversion_kwargs = {
            "input_size": input_shape,
        }

        model_type = torch.half if quantization_mode == ExportQuantizationMode.FP16 else torch.float32
        device = find_compatible_model_device_for_dtype(device, model_type)

        # This variable holds the output names of the model.
        # If postprocessing is enabled, it will be set to the output names of the postprocessing module.
        output_names: Optional[List[str]] = None

        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**prep_model_for_conversion_kwargs)

        contains_quantized_modules = check_model_contains_quantized_modules(model)

        if quantization_mode == ExportQuantizationMode.INT8:
            from super_gradients.training.utils.quantization import ptq

            model = ptq(
                model,
                selective_quantizer=selective_quantizer,
                calibration_loader=calibration_loader,
                calibration_method=calibration_method,
                calibration_batches=calibration_batches,
                calibration_percentile=calibration_percentile,
            )
        elif quantization_mode == ExportQuantizationMode.FP16:
            if contains_quantized_modules:
                raise RuntimeError("Model contains quantized modules for INT8 mode. " "FP16 quantization is not supported for such models.")
        elif quantization_mode is None and contains_quantized_modules:
            # If quantization_mode is None, but we have quantized modules in the model, we need to
            # update the quantization_mode to INT8, so that we can correctly export the model.
            quantization_mode = ExportQuantizationMode.INT8

        from super_gradients.training.models.conversion import ConvertableCompletePipelineModel

        # The model.prep_model_for_conversion will be called inside ConvertableCompletePipelineModel once more,
        # but as long as implementation of prep_model_for_conversion is idempotent, it should be fine.
        complete_model = (
            ConvertableCompletePipelineModel(model=model, pre_process=None, post_process=None, **prep_model_for_conversion_kwargs).to(device).eval()
        )

        if quantization_mode == ExportQuantizationMode.FP16:
            # For FP16 quantization, we simply can to convert the whole model to half precision
            complete_model = complete_model.half()

            if calibration_loader is not None:
                logger.warning(
                    "It seems you've passed calibration_loader to export function, but quantization_mode is set to FP16. "
                    "FP16 quantization is done by calling model.half() so you don't need to pass calibration_loader, as it will be ignored."
                )

        if engine in {ExportTargetBackend.ONNXRUNTIME}:

            onnx_export_kwargs = onnx_export_kwargs or {}
            onnx_input = torch.randn(input_shape).to(device=device, dtype=input_image_dtype)

            export_to_onnx(
                model=complete_model,
                model_input=onnx_input,
                onnx_filename=output,
                input_names=["input"],
                output_names=output_names,
                onnx_opset=onnx_export_kwargs.get("opset_version", None),
                do_constant_folding=onnx_export_kwargs.get("do_constant_folding", True),
                dynamic_axes=onnx_export_kwargs.get("dynamic_axes", None),
                keep_initializers_as_inputs=onnx_export_kwargs.get("keep_initializers_as_inputs", False),
                verbose=onnx_export_kwargs.get("verbose", False),
            )

            if onnx_simplify:
                model_opt, simplify_successful = onnxsim.simplify(output)
                if not simplify_successful:
                    raise RuntimeError(f"Failed to simplify ONNX model {output} with onnxsim. Please check the logs for details.")
                onnx.save(model_opt, output)

                logger.debug(f"Ran onnxsim.simplify on {output}")
        else:
            raise ValueError(f"Unsupported export format: {engine}. Supported formats: onnxruntime, tensorrt")

        # Cleanup memory, not sure whether it is necessary but just in case
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Add usage instructions
        usage_instructions.append(f"Model exported successfully to {output}")
        usage_instructions.append(
            f"Model expects input image of shape [{batch_size}, {2}, {input_image_channels}, {input_image_shape[0]}, {input_image_shape[1]}]"
        )
        usage_instructions.append(f"Input image dtype is {input_image_dtype}")

        usage_instructions.append("Exported model is in ONNX format and can be used with ONNXRuntime")
        usage_instructions.append("To run inference with ONNXRuntime, please use the following code snippet:")
        usage_instructions.append("")
        usage_instructions.append("    import onnxruntime")
        usage_instructions.append("    import numpy as np")

        usage_instructions.append(f'    session = onnxruntime.InferenceSession("{output}", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])')
        usage_instructions.append("    inputs = [o.name for o in session.get_inputs()]")
        usage_instructions.append("    outputs = [o.name for o in session.get_outputs()]")

        dtype_name = np.dtype(torch_dtype_to_numpy_dtype(input_image_dtype)).name
        usage_instructions.append(
            f"    example_input_batch = np.zeros(({batch_size}, {2}, {input_image_channels}, {input_image_shape[0]}, {input_image_shape[1]})).astype(np.{dtype_name})"  # noqa
        )

        usage_instructions.append("    flow_prediction = session.run(outputs, {inputs[0]: example_input_batch})")
        usage_instructions.append("")

        return OpticalFlowModelExportResult(
            input_image_channels=input_image_channels,
            input_image_dtype=input_image_dtype,
            input_image_shape=input_image_shape,
            engine=engine,
            quantization_mode=quantization_mode,
            output=output,
            usage_instructions="\n".join(usage_instructions),
        )
