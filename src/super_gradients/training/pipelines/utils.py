# from abc import ABC, abstractmethod
# from typing import Dict, Optional, Tuple, Any
#
# from super_gradients.training.models.sg_module import SgModule
# from super_gradients.training.transforms.processing import (
#     Processing,
#     ComposeProcessing,
#     DetectionPaddedRescale,
#     DetectionPadToSize,
#     ImagePermute,
#     NormalizeImage,
#     SegmentationRescale,
# )
# from super_gradients.training.models import YoloBase, PPYoloE, PPLiteSegBase, DDRNetCustom
#
#
# def get_model_image_processor(model: SgModule) -> Processing:
#     for model_class, image_processor in MODELS_PROCESSORS.items():
#         if isinstance(model, model_class):
#             return image_processor
#     raise ValueError(f"Model {model.__call__} is not supported by this pipeline.")
#
#
# # Map models classes to image processors required to run the model
# MODELS_PROCESSORS: Dict[type, Processing] = {
#     YoloBase: DetectionPaddedRescale(target_size=(640, 640), swap=(2, 0, 1)),
#     PPYoloE: ComposeProcessing(
#         [
#             DetectionPadToSize(output_size=(640, 640), pad_value=0),
#             NormalizeImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
#             ImagePermute(permutation=(2, 0, 1)),
#         ]
#     ),
#     DDRNetCustom: ComposeProcessing(
#         [
#             SegmentationRescale(output_shape=(480, 320)),
#             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     ),
# }
