import abc
import copy
import dataclasses
import gc
from typing import Any
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
from super_gradients.module_interfaces.exceptions import ModelHasNoPreprocessingParamsException
from super_gradients.module_interfaces.supports_input_shape_check import SupportsInputShapeCheck
from super_gradients.training.utils.export_utils import (
    infer_image_shape_from_model,
    infer_image_input_channels,
    infer_num_output_classes,
)
from super_gradients.training.utils.utils import infer_model_device, check_model_contains_quantized_modules, infer_model_dtype
from torch import nn, Tensor
from torch.utils.data import DataLoader

logger = get_logger(__name__)

__all__ = ["ExportableSegmentationModel", "AbstractSegmentationDecodingModule", "SegmentationModelExportResult"]


class AbstractSegmentationDecodingModule(nn.Module):
    """
    Abstract class for decoding output from semantic segmentation models to a single tensor with class indices.
    """

    @abc.abstractmethod
    def forward(self, predictions: Any) -> Tensor:
        """
        The implementation of this method must take raw predictions from the model and convert them
        to class labels tensor of [B,H,W] shape.

        :param predictions: Input predictions from the model itself.
                            Usually, this is a tensor of shape [B, C, H, W] where C is the number of classes.
                            Could also be a tuple/list of tensors.
                            In this case, the first tensor assumed to be the class predictions.

        :return: Implementation of this method must return a single tensor of shape [B, H, W] with class indices.
        """
        raise NotImplementedError(f"forward() method is not implemented for class {self.__class__.__name__}. ")

    def get_output_names(self) -> List[str]:
        """
        Returns the names of the outputs of the module.
        Usually you don't need to override this method.
        Export API uses this method internally to give meaningful names to the outputs of the exported model.

        :return: A list of output names.
        """
        return ["segmentation_mask"]


class BinarySegmentationDecodingModule(AbstractSegmentationDecodingModule):
    """
    A simple decoding module for binary segmentation.
    """

    def __init__(self, threshold: float = 0.5):
        """

        :param threshold: A threshold value for converting logits to hard binary mask.
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, predictions: Any) -> Tensor:
        """
        Convert raw predictions to binary mask.

        :param predictions: Predicted logits from the model.
               Can be a single tensor of tuple of tensors where first tensor is the predicted logits.
        :return: A tensor of long dtype with binary labels (0 or 1) of shape [B, H, W]
        """

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        return (predictions[:, 0].sigmoid() >= self.threshold).long()

    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold})"


class SemanticSegmentationDecodingModule(AbstractSegmentationDecodingModule):
    """
    A simple decoding module for multi-class segmentation.
    """

    def forward(self, predictions: Any) -> Tensor:
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        return predictions.argmax(dim=1, keepdims=False)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


@dataclasses.dataclass
class SegmentationModelExportResult:
    """
    A dataclass that holds the result of model export.
    """

    batch_size: int
    input_image_channels: int
    input_image_dtype: torch.dtype
    input_image_shape: Tuple[int, int]

    engine: ExportTargetBackend
    quantization_mode: Optional[ExportQuantizationMode]

    output: str

    usage_instructions: str = ""

    def __repr__(self):
        return self.usage_instructions


class ExportableSegmentationModel:
    """
    A mixin class that adds export functionality to the semantic segmentation models.
    Classes that inherit from this mixin must implement the following methods:
    - get_decoding_module()
    - get_preprocessing_callback()
    Providing these methods are implemented correctly, the model can be exported to ONNX or TensorRT formats
    using model.export(...) method.
    """

    def get_decoding_module(self, confidence_threshold=0.5, **kwargs) -> AbstractSegmentationDecodingModule:
        """
        Gets the decoding module for the object detection model.
        This method must be implemented by the derived class and should return
        an instance of AbstractSegmentationDecodingModule that would take raw models' outputs and
        convert them to semantic segmentation mask.

        :return: An instance of AbstractSegmentationDecodingModule
        """
        num_classes = infer_num_output_classes(self)
        if num_classes is None:
            raise NotImplementedError(
                f"Cannot infer number of output classes for {self.__class__.__name__}.\n" f"You would need to implement get_decoding_module() method manually."
            )

        if num_classes == 1:
            return BinarySegmentationDecodingModule(threshold=confidence_threshold)
        else:
            return SemanticSegmentationDecodingModule()

    def get_preprocessing_callback(self, **kwargs) -> Optional[nn.Module]:
        raise NotImplementedError(f"get_preprocessing_callback is not implemented for class {self.__class__.__name__}.")

    def export(
        self,
        output: str,
        confidence_threshold: Optional[float] = None,
        quantization_mode: Optional[ExportQuantizationMode] = None,
        selective_quantizer: Optional["SelectiveQuantizer"] = None,  # noqa
        calibration_loader: Optional[DataLoader] = None,
        calibration_method: str = "percentile",
        calibration_batches: int = 16,
        calibration_percentile: float = 99.99,
        preprocessing: Union[bool, nn.Module] = True,
        postprocessing: Union[bool, nn.Module] = True,
        postprocessing_kwargs: Optional[dict] = None,
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
        :param confidence_threshold: (float) Confidence threshold for the exported model.
               This parameter is used only for binary segmentation models (num_classes = 1).
               If None, a default value of 0.5 will be used.
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
        :param preprocessing: (bool or nn.Module)
                              If True, export a model with preprocessing that matches preprocessing params during training,
                              If False - do not use any preprocessing at all
                              If instance of nn.Module - uses given preprocessing module.
        :param postprocessing: (bool or nn.Module)
                               If True, export a model with postprocessing module obtained from model.get_post_processing_callback()
                               If False - do not use any postprocessing at all
                               If instance of nn.Module - uses given postprocessing module.
        :param postprocessing_kwargs: (dict) Optional keyword arguments for model.get_post_processing_callback(),
               used only when `postprocessing=True`.
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
        from super_gradients.conversion.preprocessing_modules import CastTensorTo

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

        input_shape = (batch_size, input_image_channels, rows, cols)

        if isinstance(model, SupportsInputShapeCheck):
            model.validate_input_shape(input_shape)

        prep_model_for_conversion_kwargs = {
            "input_size": input_shape,
        }

        model_type = torch.half if quantization_mode == ExportQuantizationMode.FP16 else torch.float32
        device = find_compatible_model_device_for_dtype(device, model_type)

        if isinstance(preprocessing, nn.Module):
            preprocessing_module = preprocessing
        elif preprocessing is True:
            try:
                preprocessing_module = model.get_preprocessing_callback()
            except ModelHasNoPreprocessingParamsException:
                raise ValueError(
                    "It looks like your model does not have dataset preprocessing params properly set.\n"
                    "This may happen if you instantiated model from scratch and not trained it yet. \n"
                    "Here are what you can do to fix this:\n"
                    "1. Manually fill up dataset processing params via model.set_dataset_processing_params(...).\n"
                    "2. Train your model first and then export it. Trainer will set_dataset_processing_params(...) for you.\n"
                    '3. Instantiate a model using pretrained weights: models.get(..., pretrained_weights="coco") \n'
                    "4. Disable preprocessing by passing model.export(..., preprocessing=False). \n"
                )
            if isinstance(preprocessing_module, nn.Sequential):
                preprocessing_module = nn.Sequential(CastTensorTo(model_type), *iter(preprocessing_module))
            else:
                preprocessing_module = nn.Sequential(CastTensorTo(model_type), preprocessing_module)
            input_image_dtype = input_image_dtype or torch.uint8
        else:
            preprocessing_module = None
            input_image_dtype = input_image_dtype or model_type

        # This variable holds the output names of the model.
        # If postprocessing is enabled, it will be set to the output names of the postprocessing module.
        if onnx_export_kwargs is not None and "output_names" in onnx_export_kwargs:
            output_names = onnx_export_kwargs.pop("output_names")
        else:
            output_names = None

        if onnx_export_kwargs is not None and "input_names" in onnx_export_kwargs:
            input_names = onnx_export_kwargs.pop("input_names")
        else:
            input_names = ["input"]

        if isinstance(postprocessing, nn.Module):
            # If a user-specified postprocessing module is provided, we will attach is to the model and not
            # attempt to attach NMS step, since we do not know what the user-specified postprocessing module does,
            # and what outputs it produces.
            postprocessing_module = postprocessing
        elif postprocessing is True:
            num_classes = infer_num_output_classes(model)
            postprocessing_kwargs = postprocessing_kwargs or {}
            if num_classes == 1:
                if confidence_threshold is None and "confidence_threshold" not in postprocessing_kwargs:
                    logger.info(
                        "A model seems to be producing a segmentation mask of one channel.\n"
                        "For binary segmentation task a confidence_threshold parameter controls the decision boundary.\n"
                        "A default value of 0.5 will be used for confidence_threshold.\n"
                        "You can override this by passing confidence_threshold to model.export(..., confidence_threshold=YOUR_VALUE)"
                    )
                    postprocessing_kwargs["confidence_threshold"] = 0.5

            postprocessing_module: AbstractSegmentationDecodingModule = model.get_decoding_module(**postprocessing_kwargs)

            output_names = postprocessing_module.get_output_names()

            dummy_input = torch.randn(input_shape).to(device=infer_model_device(model), dtype=infer_model_dtype(model))
            with torch.no_grad():
                model.eval()(dummy_input)
        else:
            postprocessing_module = None

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
            ConvertableCompletePipelineModel(
                model=model, pre_process=preprocessing_module, post_process=postprocessing_module, **prep_model_for_conversion_kwargs
            )
            .to(device)
            .eval()
        )

        if quantization_mode == ExportQuantizationMode.FP16:
            # For FP16 quantization, we simply can to convert the whole model to half precision
            complete_model = complete_model.half()

            if calibration_loader is not None:
                logger.warning(
                    "It seems you've passed calibration_loader to export function, but quantization_mode is set to FP16. "
                    "FP16 quantization is done by calling model.half() so you don't need to pass calibration_loader, as it will be ignored."
                )

        if engine in {ExportTargetBackend.ONNXRUNTIME, ExportTargetBackend.TENSORRT}:
            from super_gradients.conversion.onnx.export_to_onnx import export_to_onnx

            onnx_export_kwargs = onnx_export_kwargs or {}
            onnx_input = torch.randn(input_shape).to(device=device, dtype=input_image_dtype)

            export_to_onnx(
                model=complete_model,
                model_input=onnx_input,
                onnx_filename=output,
                input_names=input_names,
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
        usage_instructions.append(f"Model expects input image of shape [{batch_size}, {input_image_channels}, {input_image_shape[0]}, {input_image_shape[1]}]")
        usage_instructions.append(f"Input image dtype is {input_image_dtype}")

        if preprocessing:
            usage_instructions.append("Exported model already contains preprocessing (normalization) step, so you don't need to do it manually.")
            usage_instructions.append("Preprocessing steps to be applied to input image are:")
            usage_instructions.append(repr(preprocessing_module))
            usage_instructions.append("")

        if postprocessing:
            usage_instructions.append(f"Exported model contains postprocessing module {repr(postprocessing_module)}.")
            usage_instructions.append(f"Output of the model is mask of [{batch_size, *input_image_shape}] [B,H,W] shape with class indices.")
            usage_instructions.append("")

        usage_instructions.append("Exported model is in ONNX format and can be used with ONNXRuntime")
        usage_instructions.append("To run inference with ONNXRuntime, please use the following code snippet:")
        usage_instructions.append("")
        usage_instructions.append("    import onnxruntime")
        usage_instructions.append("    import numpy as np")
        usage_instructions.append("    import matplotlib.pyplot as plt")
        usage_instructions.append("    from super_gradients.training.utils.visualization.segmentation import overlay_segmentation")

        usage_instructions.append(f'    session = onnxruntime.InferenceSession("{output}", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])')
        usage_instructions.append("    inputs = [o.name for o in session.get_inputs()]")
        usage_instructions.append("    outputs = [o.name for o in session.get_outputs()]")

        dtype_name = np.dtype(torch_dtype_to_numpy_dtype(input_image_dtype)).name
        usage_instructions.append(
            f"    example_input_batch = np.zeros(({batch_size}, {input_image_channels}, {input_image_shape[0]}, {input_image_shape[1]})).astype(np.{dtype_name})"  # noqa
        )

        usage_instructions.append("    predictions = session.run(outputs, {inputs[0]: example_input_image})")
        usage_instructions.append("")

        if postprocessing is True:
            usage_instructions.append("")
            usage_instructions.append("    [segmentation_mask] = predictions")
            usage_instructions.append("    for image_index in range(segmentation_mask.shape[0]):")
            usage_instructions.append("        mask = segmentation_mask[image_index]")
            usage_instructions.append("        overlay = overlay_segmentation(")
            usage_instructions.append(f"            pred_mask=segmentation_mask, image=example_input_batch[image_index], num_classes={num_classes}")
            usage_instructions.append("        )")
            usage_instructions.append("        plt.figure(figsize=(10, 10))")
            usage_instructions.append("        plt.imshow(overlay)")
            usage_instructions.append("        plt.title('Segmentation Overlay')")
            usage_instructions.append("        plt.tight_layout()")
            usage_instructions.append("        plt.show()")

        elif postprocessing is False:
            usage_instructions.append("Model exported with postprocessing=False")
            usage_instructions.append("This means that the model produces raw predictions (logits) and you need to implement your own postprocessing step.")
            usage_instructions.append("Please refer to the documentation for the model you exported")
        elif isinstance(postprocessing, nn.Module):
            usage_instructions.append("Exported model contains a custom postprocessing step.")
            usage_instructions.append("We are unable to provide usage instructions to user-provided postprocessing module")
            usage_instructions.append("But here is the human-friendly representation of the postprocessing module:")
            usage_instructions.append(repr(postprocessing))

        return SegmentationModelExportResult(
            batch_size=batch_size,
            input_image_channels=input_image_channels,
            input_image_dtype=input_image_dtype,
            input_image_shape=input_image_shape,
            engine=engine,
            quantization_mode=quantization_mode,
            output=output,
            usage_instructions="\n".join(usage_instructions),
        )
