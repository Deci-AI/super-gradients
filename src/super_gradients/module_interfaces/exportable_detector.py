import abc
import copy
from typing import Protocol, runtime_checkable, Any
from typing import Union, Optional, List, Tuple

import onnxsim
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.conversion.onnx.nms import attach_onnx_nms
from super_gradients.conversion.tensorrt.nms import attach_tensorrt_nms
from super_gradients.training.utils.export_utils import infer_format_from_file_name, infer_image_shape_from_model
from super_gradients.training.utils.utils import infer_model_device

logger = get_logger(__name__)

__all__ = ["ExportableObjectDetectionModel", "AbstractObjectDetectionDecodingModule"]


class AbstractObjectDetectionDecodingModule(nn.Module):
    @abc.abstractmethod
    def forward(self, predictions: Any) -> Tuple[Tensor, Tensor]:
        """

        :param predictions:
        :return:
        """
        ...

    def get_output_names(self) -> List[str]:
        """
        Returns the names of the outputs of the module.

        :return:
        """
        return ["pre_nms_bboxes_xyxy", "pre_nms_scores"]

    @abc.abstractmethod
    def get_num_pre_nms_predictions(self) -> int:
        ...


@runtime_checkable
class ExportableObjectDetectionModel(Protocol):
    def get_decoding_module(self, num_pre_nms_predictions: int, **kwargs) -> AbstractObjectDetectionDecodingModule:
        """
        Gets the decoding module for the object detection model.
        This method must be implemented by the derived class and should return
        an instance of AbstractObjectDetectionDecodingModule that would take raw models' outputs and
        convert them to a tuple of two tensors (boxes, scores):
         - boxes: [B, N, 4] - All predicted boxes in (x1, y1, x2, y2) format.
         - scores: [B, N, C] - All predicted scores ([0..1] range) for each box and class.
        :return: An instance of AbstractObjectDetectionDecodingModule
        """
        ...

    def export(
        self,
        output: str,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        engine: Optional[str] = None,
        quantize: bool = False,
        calibration_loader: Optional[DataLoader] = None,
        preprocessing: Union[bool, nn.Module] = False,
        postprocessing: Union[bool, nn.Module] = True,
        postprocessing_kwargs: Optional[dict] = None,
        batch_size: int = 1,
        input_image_shape: Optional[Tuple[int, int]] = None,
        input_image_channels: Optional[int] = None,
        max_predictions_per_image: Optional[int] = None,
        onnx_export_kwargs: Optional[dict] = None,
        onnx_simplify: bool = True,
        device: Optional[torch.device] = None,
        output_predictions_format: str = "batch",
        num_pre_nms_predictions: int = 1000,
    ):
        """
        Export the model to one of supported formats. Format is inferred from the output file extension or can be
        explicitly specified via `format` argument.

        :param output: Output file name of the exported model.
        :param nms_threshold: (float) NMS threshold for the exported model.
        :param confidence_threshold: (float) Confidence threshold for the exported model.
        :param engine: Explicit specification of the inference engine. If not specified, engine is inferred from the output file extension.
                       Supported values:
                       - "onnxruntime" - export to ONNX format with ONNX runtime as inference engine.
                       Note, models that are using NMS exported in this mode ARE compatible with TRT runtime.
                       - "tensorrt" - export to ONNX format with TensorRT  as inference engine.
                       This mode enables use of efficient TensorRT NMS plugin. Note, models that are using NMS exported in this
                       mode ARE NOT COMPATIBLE with ONNX runtime.
        :param quantize: (bool) If True, export a quantized model, otherwise export a model in full precision.
        :param calibration_loader: (torch.utils.data.DataLoader) An optional data loader for calibrating a quantized model.
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
        :param max_predictions_per_image: (int) Maximum number of detections per image for the exported model.
        :param device: (torch.device) Device to use for exporting the model. If not specified, the device is inferred from the model itself.
        :param onnx_export_kwargs: (dict) Optional keyword arguments for torch.onnx.export() function.
        :param onnx_simplify: (bool) If True, apply onnx-simplifier to the exported model.
        :param output_predictions_format: (str) Format of the output predictions after NMS. Supported values: batch, flat.
        :param num_pre_nms_predictions: (int) Number of predictions to keep before NMS.
        :return:
        """
        if not isinstance(self, nn.Module):
            raise TypeError(f"Export is only supported for torch.nn.Module. Got type {type(self)}")

        device: torch.device = device or infer_model_device(self)
        model: nn.Module = copy.deepcopy(self).to(device).eval()

        engine: str = engine or infer_format_from_file_name(output)
        if engine is None:
            raise ValueError(
                "Export format is not specified and cannot be inferred from the output file name. "
                "Please specify the format explicitly: model.export(..., format='onnx|coreml')"
            )

        input_image_shape: Tuple[int, int] = input_image_shape or infer_image_shape_from_model(model)
        if input_image_shape is None:
            raise ValueError(
                "Image shape is not specified and cannot be inferred from the model. "
                "Please specify the image shape explicitly: model.export(..., image_shape=(height, width))"
            )

        try:
            rows, cols = input_image_shape
        except ValueError:
            raise ValueError(f"Image shape must be a tuple of two integers (height, width), got {input_image_shape} instead")

        input_image_channels = input_image_channels or 3

        input_shape = (batch_size, input_image_channels, rows, cols)
        prep_model_for_conversion_kwargs = {
            "input_size": input_shape,
        }

        if isinstance(preprocessing, nn.Module):
            pass
        elif preprocessing is True:
            preprocessing = self.get_preprocessing_callback()
        else:
            preprocessing = None

        # This variable holds the output names of the model.
        # If postprocessing is enabled, it will be set to the output names of the postprocessing module.
        output_names: Optional[List[str]] = None

        if isinstance(postprocessing, nn.Module):
            # If a user-specified postprocessing module is provided, we will attach is to the model and not
            # attempt to attach NMS step, since we do not know what the user-specified postprocessing module does,
            # and what outputs it produces.
            attach_nms_postprocessing = False
        elif postprocessing is True:
            attach_nms_postprocessing = True
            postprocessing_kwargs = postprocessing_kwargs or {}
            postprocessing_kwargs["num_pre_nms_predictions"] = num_pre_nms_predictions
            postprocessing: AbstractObjectDetectionDecodingModule = model.get_decoding_module(**postprocessing_kwargs)

            output_names = postprocessing.get_output_names()
            num_pre_nms_predictions = postprocessing.num_pre_nms_predictions
            max_predictions_per_image = max_predictions_per_image or num_pre_nms_predictions

            nms_threshold = nms_threshold or getattr(model, "_default_nms_iou", None)
            if nms_threshold is None:
                raise ValueError(
                    "nms_threshold is not specified and cannot be inferred from the model. "
                    "Please specify the nms_threshold explicitly: model.export(..., nms_threshold=0.5)"
                )

            confidence_threshold = confidence_threshold or getattr(model, "_default_nms_conf", None)
            if confidence_threshold is None:
                raise ValueError(
                    "confidence_threshold is not specified and cannot be inferred from the model. "
                    "Please specify the confidence_threshold explicitly: model.export(..., confidence_threshold=0.5)"
                )

            if max_predictions_per_image > num_pre_nms_predictions:
                raise ValueError(
                    f"max_predictions_per_image={max_predictions_per_image} is greater than "
                    f"num_pre_nms_predictions={num_pre_nms_predictions}. "
                    f"Please specify max_predictions_per_image <= {num_pre_nms_predictions}."
                )
        else:
            attach_nms_postprocessing = False
            postprocessing = None

        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**prep_model_for_conversion_kwargs)

        if quantize:
            logger.debug("Quantizing model")
            from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer
            from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
            from pytorch_quantization import nn as quant_nn

            q_util = SelectiveQuantizer(
                default_quant_modules_calibrator_weights="max",
                default_quant_modules_calibrator_inputs="histogram",
                default_per_channel_quant_weights=True,
                default_learn_amax=False,
                verbose=True,
            )
            q_util.quantize_module(model)

            if calibration_loader:
                logger.debug("Calibrating model")
                calibrator = QuantizationCalibrator(verbose=True)
                calibrator.calibrate_model(
                    model,
                    method="percentile",
                    calib_data_loader=calibration_loader,
                    num_calib_batches=16,
                    percentile=99.99,
                )
                logger.debug("Calibrating model complete")

        from super_gradients.training.models.conversion import ConvertableCompletePipelineModel

        # The model.prep_model_for_conversion will be called inside ConvertableCompletePipelineModel once more,
        # but as long as implementation of prep_model_for_conversion is idempotent, it should be fine.
        complete_model = ConvertableCompletePipelineModel(
            model=model, pre_process=preprocessing, post_process=postprocessing, **prep_model_for_conversion_kwargs
        )

        if engine in {"onnxruntime", "tensorrt"}:
            onnx_export_kwargs = onnx_export_kwargs or {}

            if quantize:
                use_fb_fake_quant_state = quant_nn.TensorQuantizer.use_fb_fake_quant
                quant_nn.TensorQuantizer.use_fb_fake_quant = True

            try:
                with torch.no_grad():
                    onnx_input = torch.randn(input_shape, device=device)
                    logger.debug("Exporting model to ONNX")
                    logger.debug("ONNX input shape: %s", input_shape)
                    logger.debug("ONNX output names: %s", output_names)
                    torch.onnx.export(model=complete_model, args=onnx_input, f=output, output_names=output_names, **onnx_export_kwargs)

                # Stitch ONNX graph with NMS postprocessing
                if attach_nms_postprocessing:
                    if engine == "tensorrt":

                        if onnx_simplify:
                            # If TRT engine is used, we need to run onnxsim.simplify BEFORE attaching NMS,
                            # because EfficientNMS_TRT is not supported by onnxsim and would lead to a runtime error.
                            onnxsim.simplify(output)
                            logger.debug(f"Ran onnxsim.simplify on model {output}")
                            # Disable onnx_simplify to avoid running it twice.
                            onnx_simplify = False

                        nms_attach_method = attach_tensorrt_nms

                        if output_predictions_format == "flat":
                            logger.warning(
                                "Support of flat predictions format in TensorRT is experimental and may not work on all versions of TensorRT. "
                                "We recommend using TensorRT 8.4.1 or newer. On older versions this format will not work. "
                                "If you encountering issues loading exported model in TensorRT, please try upgrading TensorRT to latest version. "
                                "Alternatively, you can export the model to output predictions in batch format by "
                                "specifying output_predictions_format='batch'"
                            )
                    elif engine == "onnxruntime":
                        nms_attach_method = attach_onnx_nms
                    else:
                        raise KeyError(f"Unsupported engine: {engine}")

                    nms_attach_method(
                        onnx_model_path=output,
                        output_onnx_model_path=output,
                        num_pre_nms_predictions=num_pre_nms_predictions,
                        max_predictions_per_image=max_predictions_per_image,
                        nms_threshold=nms_threshold,
                        confidence_threshold=confidence_threshold,
                        batch_size=batch_size,
                        output_predictions_format=output_predictions_format,
                    )

                if onnx_simplify:
                    onnxsim.simplify(output)
                    logger.debug(f"Ran onnxsim.simplify on {output}")
            finally:
                if quantize:
                    # Restore functions of quant_nn back as expected
                    quant_nn.TensorQuantizer.use_fb_fake_quant = use_fb_fake_quant_state

        else:
            raise ValueError(f"Unsupported export format: {engine}. Supported formats: onnxruntime, tensorrt")
