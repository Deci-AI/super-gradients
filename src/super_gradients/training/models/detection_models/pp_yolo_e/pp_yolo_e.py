import copy
from typing import Union, Optional, List, Tuple
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.conversion.onnx.nms import attach_onnx_nms
from super_gradients.conversion.tensorrt.nms import attach_tensorrt_nms
from super_gradients.modules import RepVGGBlock
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.models.detection_models.csp_resnet import CSPResNetBackbone
from super_gradients.training.models.detection_models.pp_yolo_e.pan import PPYoloECSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from super_gradients.training.utils import HpmStruct
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.pp_yolo_e.post_prediction_callback import PPYoloEPostPredictionCallback, DetectionPostPredictionCallback
from super_gradients.training.utils.export_utils import infer_format_from_file_name, infer_image_shape_from_model
from super_gradients.training.utils.predict import ImagesDetectionPrediction
from super_gradients.training.pipelines.pipelines import DetectionPipeline
from super_gradients.training.processing.processing import Processing
from super_gradients.training.utils.media.image import ImageSource
from super_gradients.training.utils.utils import infer_model_device

logger = get_logger(__name__)


class PPYoloEPostprocessingModuleForTRT(nn.Module):
    """
    Decoding module for TRT NMS
    Takes in the output of the model and returns the decoded boxes in the format Tuple[Tensor, Tensor]
    * boxes [batch_size, number_boxes, 4], boxes are in format (x1, y1, x2, y2)
    * scores [batch_size, number_boxes, number_classes]
    """

    __constants__ = ["pre_nms_top_k", "multi_label_per_box"]

    def __init__(
        self,
        pre_nms_top_k: int = 300,
        multi_label_per_box: bool = False,
    ):
        super().__init__()
        self.pre_nms_top_k = pre_nms_top_k
        self.multi_label_per_box = multi_label_per_box

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """

        :param inputs: Tuple [Tensor, Tensor]
            * boxes [B, N, 4], boxes are in format (x1, y1, x2, y2)
            * scores [B, N, C]
        :return:
            * boxes [B, Nout, 4], boxes are in format (x1, y1, x2, y2)
            * scores [B, Nout, C]
        """
        pred_bboxes, pred_scores = inputs

        nms_top_k = self.pre_nms_top_k

        if self.multi_label_per_box:
            pred_cls_conf, _ = torch.max(pred_scores, dim=2)
            topk_candidates = torch.topk(pred_cls_conf, dim=1, k=nms_top_k, largest=True)
        else:
            pred_cls_conf, _ = torch.max(pred_scores, dim=2)
            topk_candidates = torch.topk(pred_cls_conf, dim=1, k=nms_top_k, largest=True)

        offsets = nms_top_k * torch.arange(pred_cls_conf.size(0), device=pred_cls_conf.device)
        flat_indices = topk_candidates.indices + offsets.reshape(pred_cls_conf.size(0), 1)
        flat_indices = torch.flatten(flat_indices)

        output_pred_bboxes = pred_bboxes.reshape(-1, pred_bboxes.size(2))[flat_indices, :].reshape(pred_bboxes.size(0), nms_top_k, pred_bboxes.size(2))
        output_pred_scores = pred_scores.reshape(-1, pred_scores.size(2))[flat_indices, :].reshape(pred_scores.size(0), nms_top_k, pred_scores.size(2))

        return output_pred_bboxes, output_pred_scores

    def get_output_names(self):
        return ["pre_nms_bboxes_xyxy", "pre_nms_scores"]


class PPYoloE(SgModule):
    def __init__(self, arch_params):
        super().__init__()
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()

        self.backbone = CSPResNetBackbone(**arch_params["backbone"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.neck = PPYoloECSPPAN(**arch_params["neck"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.head = PPYOLOEHead(**arch_params["head"], width_mult=arch_params["width_mult"], num_classes=arch_params["num_classes"])

        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        self._default_nms_iou: Optional[float] = None
        self._default_nms_conf: Optional[float] = None

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> DetectionPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        class_names: Optional[List[str]] = None,
        image_processor: Optional[Processing] = None,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:     (Optional) Names of the dataset the model was trained on.
        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param iou:             (Optional) IoU threshold for the nms algorithm
        :param conf:            (Optional) Below the confidence threshold, prediction are discarded
        """
        self._class_names = class_names or self._class_names
        self._image_processor = image_processor or self._image_processor
        self._default_nms_iou = iou or self._default_nms_iou
        self._default_nms_conf = conf or self._default_nms_conf

    @lru_cache(maxsize=1)
    def _get_pipeline(self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True) -> DetectionPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        if None in (self._class_names, self._image_processor, self._default_nms_iou, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        iou = iou or self._default_nms_iou
        conf = conf or self._default_nms_conf

        pipeline = DetectionPipeline(
            model=self,
            image_processor=self._image_processor,
            post_prediction_callback=self.get_post_prediction_callback(iou=iou, conf=conf),
            class_names=self._class_names,
        )
        return pipeline

    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
    ) -> ImagesDetectionPrediction:
        """Predict an image or a list of images.

        :param images:      Images to predict.
        :param iou:         (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:        (Optional) Below the confidence threshold, prediction are discarded.
                            If None, the default value associated to the training is used.
        :param batch_size:  Maximum number of images to process at the same time.
        :param fuse_model:  If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model)
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(self, iou: Optional[float] = None, conf: Optional[float] = None, fuse_model: bool = True):
        """Predict using webcam.

        :param iou:     (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:    (Optional) Below the confidence threshold, prediction are discarded.
                        If None, the default value associated to the training is used.
        :param fuse_model: If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        """
        pipeline = self._get_pipeline(iou=iou, conf=conf, fuse_model=fuse_model)
        pipeline.predict_webcam()

    def train(self, mode: bool = True):
        self._get_pipeline.cache_clear()
        torch.cuda.empty_cache()
        return super().train(mode)

    def forward(self, x: Tensor):
        features = self.backbone(x)
        features = self.neck(features)
        return self.head(features)

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        """
        Prepare the model to be converted to ONNX or other frameworks.
        Typically, this function will freeze the size of layers which is otherwise flexible, replace some modules
        with convertible substitutes and remove all auxiliary or training related parts.
        :param input_size: [H,W]
        """
        self.head.cache_anchors(input_size)

        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.fuse_block_residual_branches()

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.head = new_head
        else:
            self.head.replace_num_classes(new_num_classes)

    def get_postprocessing_module(self):
        return PPYoloEPostprocessingModuleForTRT()

    def export(
        self,
        output: str,
        engine: Optional[str] = None,
        quantize: bool = False,
        calibration_loader: Optional[DataLoader] = None,
        preprocessing: Union[bool, nn.Module] = True,
        postprocessing: Union[bool, nn.Module] = True,
        postprocessing_kwargs: Optional[dict] = None,
        batch_size: int = 1,
        image_shape: Optional[Tuple[int, int]] = None,
        onnx_opset_version: Optional[int] = None,
        onnx_export_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        output_predictions_format: Optional[str] = None,
    ):
        """
        Export the model to one of supported formats. Format is inferred from the output file extension or can be
        explicitly specified via `format` argument.

        :param output: Output file name of the exported model.
        :param engine: Explicit specification of the inference engine. If not specified, engine is inferred from the output file extension.
                       Supported values:
                       - "onnx" - export to ONNX format with ONNX runtime as inference engine.
                       Note, models that are using NMS exported in this mode may not compatible with TRT runtime.
                       - "tensorrt" - export to ONNX format with TensorRT  as inference engine.
                       This mode enables use of efficient TensorRT NMS plugin. Note, models that are using NMS exported in this
                       mode may not compatible with ONNX runtime.
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
        :param include_nms: (bool) If True, export a model with NMS postprocessing, otherwise export a model
               without NMS (model will output raw predictions without decoding and NMS).
        :param batch_size: (int) Batch size for the exported model.
        :param image_shape: (tuple) Input image shape (height, width) for the exported model.
               If None, the function will infer the image shape from the model's preprocessing params.
        :param nms_threshold: (float) NMS threshold for the exported model.
        :param confidence_threshold: (float) Confidence threshold for the exported model.
        :param max_detections: (int) Maximum number of detections for the exported model.
        :param onnx_opset_version: (int) ONNX opset version for the exported model.
               If not specified, the default opset is used (defined by torch version installed).
        :param device: (torch.device) Device to use for exporting the model. If not specified, the device is inferred from the model itself.
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

        image_shape: Tuple[int, int] = image_shape or infer_image_shape_from_model(model)
        if image_shape is None:
            raise ValueError(
                "Image shape is not specified and cannot be inferred from the model. "
                "Please specify the image shape explicitly: model.export(..., image_shape=(height, width))"
            )

        try:
            rows, cols = image_shape
        except ValueError:
            raise ValueError(f"Image shape must be a tuple of two integers (height, width), got {image_shape} instead")

        input_shape = (batch_size, 3, rows, cols)
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
            pass
        elif postprocessing is True:
            # if batch_size != 1:
            #     raise ValueError(
            #         "Postprocessing is not supported for batch size > 1. " "Please specify postprocessing=False to export a model without postprocessing."
            #     )
            postprocessing_kwargs = postprocessing_kwargs or {}
            postprocessing = self.get_postprocessing_module(**postprocessing_kwargs)
            output_names = postprocessing.get_output_names()
        else:
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

        complete_model = ConvertableCompletePipelineModel(self, preprocessing, postprocessing, **prep_model_for_conversion_kwargs)

        if engine in {"onnx", "tensorrt"}:
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
                    torch.onnx.export(
                        model=complete_model, args=onnx_input, f=output, opset_version=onnx_opset_version, output_names=output_names, **onnx_export_kwargs
                    )

                # Stitch ONNX graph with NMS postprocessing
                if postprocessing:
                    if engine == "tensorrt":
                        nms_attach_method = attach_tensorrt_nms
                        output_predictions_format = output_predictions_format or "batched"
                    elif engine == "onnx":
                        nms_attach_method = attach_onnx_nms
                        output_predictions_format = output_predictions_format or "flat"
                    else:
                        raise KeyError(f"Unsupported engine: {engine}")

                    nms_attach_method(
                        onnx_model_path=output,
                        output_onnx_model_path=output,
                        max_predictions_per_image=100,
                        nms_threshold=0.5,
                        confidence_threshold=0.6,
                        batch_size=batch_size,
                        output_predictions_format=output_predictions_format,
                    )

            finally:
                if quantize:
                    # Restore functions of quant_nn back as expected
                    quant_nn.TensorQuantizer.use_fb_fake_quant = use_fb_fake_quant_state

        else:
            raise ValueError(f"Unsupported export format: {engine}")


@register_model(Models.PP_YOLOE_S)
class PPYoloE_S(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_s_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_M)
class PPYoloE_M(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_m_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_L)
class PPYoloE_L(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_l_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)


@register_model(Models.PP_YOLOE_X)
class PPYoloE_X(PPYoloE):
    def __init__(self, arch_params):
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()
        arch_params = get_arch_params("ppyoloe_x_arch_params", overriding_params=arch_params)
        super().__init__(arch_params)
