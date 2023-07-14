import copy
from typing import Union, Optional, Tuple

import torch
import torchvision.ops
from omegaconf import DictConfig
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.object_names import Models
from super_gradients.common.registry import register_model
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils import get_param
from super_gradients.training.utils.export_utils import infer_format_from_file_name, infer_image_shape_from_model
from super_gradients.training.utils.utils import HpmStruct, infer_model_device
from torch import nn, Tensor
from torch.utils.data import DataLoader

logger = get_logger(__name__)


class YoloNASPostprocessingModule(nn.Module):
    def __init__(
        self,
        nms_top_k=100,
        nms_threshold=0.6,
    ):
        super().__init__()
        self.multi_label_per_box = False
        self.score_threshold = 0.5
        self.nms_top_k = nms_top_k
        self.nms_threshold = nms_threshold

    def forward(self, inputs: Tuple[Tensor, Tensor]):
        pred_bboxes, pred_scores = inputs
        pred_scores = pred_scores[0]
        pred_bboxes = pred_bboxes[0]

        # Filter all predictions by self.score_threshold
        if self.multi_label_per_box:
            i, j = (pred_scores > self.score_threshold).nonzero(as_tuple=False).T
            pred_bboxes = pred_bboxes[i]
            pred_cls_conf = pred_scores[i, j]
            pred_cls_label = j[:]
        else:
            pred_cls_conf, pred_cls_label = torch.max(pred_scores, dim=1)
            conf_mask = pred_cls_conf >= self.score_threshold

            pred_cls_conf = pred_cls_conf[conf_mask]
            pred_cls_label = pred_cls_label[conf_mask]
            pred_bboxes = pred_bboxes[conf_mask, :]

        nms_top_k = pred_cls_conf.size(0).clamp_max(self.nms_top_k)

        topk_candidates = torch.topk(pred_cls_conf, k=nms_top_k, largest=True)
        pred_cls_conf = pred_cls_conf[topk_candidates.indices]
        pred_cls_label = pred_cls_label[topk_candidates.indices]
        pred_bboxes = pred_bboxes[topk_candidates.indices, :]

        # NMS
        idx_to_keep = torchvision.ops.boxes.batched_nms(boxes=pred_bboxes, scores=pred_cls_conf, idxs=pred_cls_label, iou_threshold=self.nms_threshold)

        pred_cls_conf = pred_cls_conf[idx_to_keep].unsqueeze(-1)
        pred_cls_label = pred_cls_label[idx_to_keep].unsqueeze(-1)
        pred_bboxes = pred_bboxes[idx_to_keep, :]

        #  nx6 (x1, y1, x2, y2, confidence, class) in pixel units
        final_boxes = torch.cat([pred_bboxes, pred_cls_conf, pred_cls_label], dim=1)  # [N,6]

        return final_boxes


class YoloNAS(CustomizableDetector):
    def get_post_prediction_callback(conf: float, iou: float) -> PPYoloEPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    def get_postprocessing_module(self):
        return YoloNASPostprocessingModule()

    def export(
        self,
        output: str,
        format: Optional[str] = None,
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
    ):
        """
        Export the model to one of supported formats. Format is inferred from the output file extension or can be
        explicitly specified via `format` argument.

        :param output: Output file name of the exported model.
        :param format: Explicit specification of export format. If not specified, format is inferred from the output file extension.
                       Currently only onnx and coreml formats are supported.
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

        format: str = format or infer_format_from_file_name(output)
        if format is None:
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

        if isinstance(postprocessing, nn.Module):
            pass
        elif postprocessing is True:
            postprocessing_kwargs = postprocessing_kwargs or {}
            postprocessing = self.get_postprocessing_module(**postprocessing_kwargs)
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

        if format == "onnx":
            onnx_export_kwargs = onnx_export_kwargs or {}

            if quantize:
                use_fb_fake_quant_state = quant_nn.TensorQuantizer.use_fb_fake_quant
                quant_nn.TensorQuantizer.use_fb_fake_quant = True

            try:
                with torch.no_grad():
                    onnx_input = torch.randn(input_shape, device=device)
                    torch.onnx.export(model=complete_model, args=onnx_input, f=output, opset_version=onnx_opset_version, **onnx_export_kwargs)

                    # node = onnx.helper.make_node(
                    #     "NonMaxSuppression",
                    #     inputs=[
                    #         "raw_boxes",
                    #         "raw_scores",
                    #         "max_output_boxes_per_class",
                    #         "iou_threshold",
                    #         "score_threshold",
                    #     ],
                    #     outputs=["selected_indices"],
                    #     center_point_box=1,
                    # )

            finally:
                if quantize:
                    # Restore functions of quant_nn back as expected
                    quant_nn.TensorQuantizer.use_fb_fake_quant = use_fb_fake_quant_state

        else:
            raise ValueError(f"Unsupported export format: {format}")


@register_model(Models.YOLO_NAS_S)
class YoloNAS_S(YoloNAS):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_s_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> PPYoloEPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_M)
class YoloNAS_M(YoloNAS):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_m_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> PPYoloEPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @property
    def num_classes(self):
        return self.heads.num_classes


@register_model(Models.YOLO_NAS_L)
class YoloNAS_L(YoloNAS):
    def __init__(self, arch_params: Union[HpmStruct, DictConfig]):
        default_arch_params = get_arch_params("yolo_nas_l_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(default_arch_params))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            backbone=merged_arch_params.backbone,
            neck=merged_arch_params.neck,
            heads=merged_arch_params.heads,
            num_classes=get_param(merged_arch_params, "num_classes", None),
            in_channels=get_param(merged_arch_params, "in_channels", 3),
            bn_momentum=get_param(merged_arch_params, "bn_momentum", None),
            bn_eps=get_param(merged_arch_params, "bn_eps", None),
            inplace_act=get_param(merged_arch_params, "inplace_act", None),
        )

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> PPYoloEPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    @property
    def num_classes(self):
        return self.heads.num_classes
