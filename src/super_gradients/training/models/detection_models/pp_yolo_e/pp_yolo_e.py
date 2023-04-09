from typing import Union, Optional, List

from torch import Tensor

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.modules import RepVGGBlock
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.models.detection_models.csp_resnet import CSPResNetBackbone
from super_gradients.training.models.detection_models.pp_yolo_e.pan import CustomCSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from super_gradients.training.utils import HpmStruct
from super_gradients.training.models.arch_params_factory import get_arch_params
from super_gradients.training.models.detection_models.pp_yolo_e.post_prediction_callback import PPYoloEPostPredictionCallback, DetectionPostPredictionCallback
from super_gradients.training.models.results import DetectionResults
from super_gradients.training.pipelines.pipelines import DetectionPipeline
from super_gradients.training.transforms.processing import Processing
from super_gradients.training.utils.media.videos import visualize_video


class PPYoloE(SgModule):
    def __init__(self, arch_params):
        super().__init__()
        if isinstance(arch_params, HpmStruct):
            arch_params = arch_params.to_dict()

        self.backbone = CSPResNetBackbone(**arch_params["backbone"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.neck = CustomCSPPAN(**arch_params["neck"], depth_mult=arch_params["depth_mult"], width_mult=arch_params["width_mult"])
        self.head = PPYOLOEHead(**arch_params["head"], width_mult=arch_params["width_mult"], num_classes=arch_params["num_classes"])

        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None

    @staticmethod
    def get_post_prediction_callback(conf: float, iou: float) -> DetectionPostPredictionCallback:
        return PPYoloEPostPredictionCallback(score_threshold=conf, nms_threshold=iou, nms_top_k=1000, max_predictions=300)

    def set_dataset_processing_params(self, class_names: Optional[List[str]], image_processor: Optional[Processing]) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:     (Optional) Names of the dataset the model was trained on.
        :param image_processor: (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        """
        self._class_names = class_names or self._class_names
        self._image_processor = image_processor or self._image_processor

    def _get_pipeline(self, iou: float, conf: float) -> DetectionPipeline:
        if self._class_names is None or self._image_processor is None:
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n" "Please call `model.set_dataset_processing_params(...)` first."
            )

        pipeline = DetectionPipeline(
            model=self,
            image_processor=self._image_processor,
            post_prediction_callback=self.get_post_prediction_callback(iou=iou, conf=conf),
            class_names=self._class_names,
        )
        return pipeline

    def predict(self, images, iou: float = 0.65, conf: float = 0.01) -> DetectionResults:
        pipeline = self._get_pipeline(iou=iou, conf=conf)
        return pipeline.predict_images(images)  # type: ignore

    def predict_image_folder(self, image_folder_path: str, output_folder_path: str, iou: float = 0.65, conf: float = 0.01, batch_size: Optional[int] = 32):
        pipeline = self._get_pipeline(iou=iou, conf=conf)
        pipeline.predict_image_folder(image_folder_path=image_folder_path, output_folder_path=output_folder_path, batch_size=batch_size)

    def predict_video(
        self, video_path: str, iou: float = 0.65, conf: float = 0.01, output_video_path: str = None, batch_size: Optional[int] = 32, visualize: bool = False
    ):
        pipeline = self._get_pipeline(iou=iou, conf=conf)
        pipeline.predict_video(video_path=video_path, output_video_path=output_video_path, batch_size=batch_size)
        if visualize:
            visualize_video(output_video_path)

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
