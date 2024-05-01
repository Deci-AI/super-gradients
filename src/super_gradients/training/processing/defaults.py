from super_gradients.training.datasets.datasets_conf import (
    COCO_DETECTION_CLASSES_LIST,
    DOTA2_DEFAULT_CLASSES_LIST,
    IMAGENET_CLASSES,
    CITYSCAPES_DEFAULT_SEGMENTATION_CLASSES_LIST,
)

from .obb import OBBDetectionCenterPadding, OBBDetectionLongestMaxSizeRescale
from .processing import (
    ComposeProcessing,
    ReverseImageChannels,
    DetectionLongestMaxSizeRescale,
    DetectionBottomRightPadding,
    ImagePermute,
    DetectionRescale,
    NormalizeImage,
    DetectionCenterPadding,
    StandardizeImage,
    KeypointsLongestMaxSizeRescale,
    KeypointsBottomRightPadding,
    CenterCrop,
    Resize,
    SegmentationResizeWithPadding,
    SegmentationRescale,
    SegmentationPadShortToCropSize,
)


def default_yolox_coco_processing_params() -> dict:
    """Processing parameters commonly used for training YoloX on COCO dataset.
    TODO: remove once we load it from the checkpoint
    """

    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),
            DetectionLongestMaxSizeRescale((640, 640)),
            DetectionBottomRightPadding((640, 640), 114),
            ImagePermute((2, 0, 1)),
        ]
    )

    params = dict(
        class_names=COCO_DETECTION_CLASSES_LIST,
        image_processor=image_processor,
        iou=0.65,
        conf=0.1,
    )
    return params


def default_ppyoloe_coco_processing_params() -> dict:
    """Processing parameters commonly used for training PPYoloE on COCO dataset.
    TODO: remove once we load it from the checkpoint
    """

    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),
            DetectionRescale(output_shape=(640, 640)),
            NormalizeImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    params = dict(
        class_names=COCO_DETECTION_CLASSES_LIST,
        image_processor=image_processor,
        iou=0.65,
        conf=0.5,
    )
    return params


def default_yolo_nas_coco_processing_params() -> dict:
    """Processing parameters commonly used for training YoloNAS on COCO dataset.
    TODO: remove once we load it from the checkpoint
    """

    image_processor = ComposeProcessing(
        [
            DetectionLongestMaxSizeRescale(output_shape=(636, 636)),
            DetectionCenterPadding(output_shape=(640, 640), pad_value=114),
            StandardizeImage(max_value=255.0),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    params = dict(
        class_names=COCO_DETECTION_CLASSES_LIST,
        image_processor=image_processor,
        iou=0.7,
        conf=0.25,
    )
    return params


def default_yolo_nas_r_dota_processing_params() -> dict:
    """Processing parameters commonly used for training YoloNAS on COCO dataset."""

    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),  # Model trained on BGR images
            OBBDetectionLongestMaxSizeRescale(output_shape=(1024, 1024)),
            OBBDetectionCenterPadding(output_shape=(1024, 1024), pad_value=114),
            StandardizeImage(max_value=255.0),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    params = dict(
        class_names=DOTA2_DEFAULT_CLASSES_LIST,
        image_processor=image_processor,
        iou=0.7,
        conf=0.25,
    )
    return params


def default_dekr_coco_processing_params() -> dict:
    """Processing parameters commonly used for training DEKR on COCO dataset."""

    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),
            KeypointsLongestMaxSizeRescale(output_shape=(640, 640)),
            KeypointsBottomRightPadding(output_shape=(640, 640), pad_value=127),
            StandardizeImage(max_value=255.0),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    edge_links = [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 6],
        [5, 7],
        [5, 11],
        [6, 8],
        [6, 12],
        [7, 9],
        [8, 10],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],
    ]

    edge_colors = [
        (214, 39, 40),  # Nose -> LeftEye
        (148, 103, 189),  # Nose -> RightEye
        (44, 160, 44),  # LeftEye -> RightEye
        (140, 86, 75),  # LeftEye -> LeftEar
        (227, 119, 194),  # RightEye -> RightEar
        (127, 127, 127),  # LeftEar -> LeftShoulder
        (188, 189, 34),  # RightEar -> RightShoulder
        (127, 127, 127),  # Shoulders
        (188, 189, 34),  # LeftShoulder -> LeftElbow
        (140, 86, 75),  # LeftTorso
        (23, 190, 207),  # RightShoulder -> RightElbow
        (227, 119, 194),  # RightTorso
        (31, 119, 180),  # LeftElbow -> LeftArm
        (255, 127, 14),  # RightElbow -> RightArm
        (148, 103, 189),  # Waist
        (255, 127, 14),  # Left Hip -> Left Knee
        (214, 39, 40),  # Right Hip -> Right Knee
        (31, 119, 180),  # Left Knee -> Left Ankle
        (44, 160, 44),  # Right Knee -> Right Ankle
    ]

    keypoint_colors = [
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
    ]
    params = dict(image_processor=image_processor, conf=0.05, edge_links=edge_links, edge_colors=edge_colors, keypoint_colors=keypoint_colors)
    return params


def default_yolo_nas_pose_coco_processing_params():
    image_processor = ComposeProcessing(
        [
            ReverseImageChannels(),
            KeypointsLongestMaxSizeRescale(output_shape=(640, 640)),
            KeypointsBottomRightPadding(output_shape=(640, 640), pad_value=127),
            StandardizeImage(max_value=255.0),
            ImagePermute(permutation=(2, 0, 1)),
        ]
    )

    edge_links = [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 6],
        [5, 7],
        [5, 11],
        [6, 8],
        [6, 12],
        [7, 9],
        [8, 10],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],
    ]

    edge_colors = [
        (214, 39, 40),  # Nose -> LeftEye
        (148, 103, 189),  # Nose -> RightEye
        (44, 160, 44),  # LeftEye -> RightEye
        (140, 86, 75),  # LeftEye -> LeftEar
        (227, 119, 194),  # RightEye -> RightEar
        (127, 127, 127),  # LeftEar -> LeftShoulder
        (188, 189, 34),  # RightEar -> RightShoulder
        (127, 127, 127),  # Shoulders
        (188, 189, 34),  # LeftShoulder -> LeftElbow
        (140, 86, 75),  # LeftTorso
        (23, 190, 207),  # RightShoulder -> RightElbow
        (227, 119, 194),  # RightTorso
        (31, 119, 180),  # LeftElbow -> LeftArm
        (255, 127, 14),  # RightElbow -> RightArm
        (148, 103, 189),  # Waist
        (255, 127, 14),  # Left Hip -> Left Knee
        (214, 39, 40),  # Right Hip -> Right Knee
        (31, 119, 180),  # Left Knee -> Left Ankle
        (44, 160, 44),  # Right Knee -> Right Ankle
    ]

    keypoint_colors = [
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
        (31, 119, 180),
        (148, 103, 189),
    ]
    params = dict(image_processor=image_processor, conf=0.5, edge_links=edge_links, edge_colors=edge_colors, keypoint_colors=keypoint_colors)
    return params


def default_imagenet_processing_params() -> dict:
    """Processing parameters commonly used for training resnet on Imagenet dataset."""
    image_processor = ComposeProcessing(
        [Resize(size=256), CenterCrop(size=224), StandardizeImage(), NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ImagePermute()]
    )
    params = dict(
        class_names=IMAGENET_CLASSES,
        image_processor=image_processor,
    )
    return params


def default_vit_imagenet_processing_params() -> dict:
    """Processing parameters used by ViT for training resnet on Imagenet dataset."""
    image_processor = ComposeProcessing(
        [Resize(size=256), CenterCrop(size=224), StandardizeImage(), NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ImagePermute()]
    )
    params = dict(
        class_names=IMAGENET_CLASSES,
        image_processor=image_processor,
    )
    return params


def default_cityscapes_processing_params(scale: float = 1) -> dict:
    """Processing parameters commonly used for training segmentation models on Cityscapes dataset."""
    image_processor = ComposeProcessing(
        [
            SegmentationResizeWithPadding(output_shape=(int(1024 * scale), int(2048 * scale)), pad_value=0),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            StandardizeImage(),
            ImagePermute(),
        ]
    )
    params = dict(
        class_names=CITYSCAPES_DEFAULT_SEGMENTATION_CLASSES_LIST,
        image_processor=image_processor,
    )
    return params


def default_segformer_cityscapes_processing_params() -> dict:
    """Processing parameters commonly used for training Segformer on Cityscapes dataset."""
    image_processor = ComposeProcessing(
        [
            SegmentationRescale(long_size=1024),
            SegmentationPadShortToCropSize(crop_size=(1024, 2048), fill_image=0),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            StandardizeImage(),
            ImagePermute(),
        ]
    )
    params = dict(
        class_names=CITYSCAPES_DEFAULT_SEGMENTATION_CLASSES_LIST,
        image_processor=image_processor,
    )
    return params


def get_pretrained_processing_params(model_name: str, pretrained_weights: str) -> dict:
    """Get the processing parameters for a pretrained model.
    TODO: remove once we load it from the checkpoint
    """
    if pretrained_weights == "coco":
        if "yolox" in model_name:
            return default_yolox_coco_processing_params()
        elif "ppyoloe" in model_name:
            return default_ppyoloe_coco_processing_params()
        elif "yolo_nas" in model_name:
            return default_yolo_nas_coco_processing_params()

    if pretrained_weights == "coco_pose" and model_name in ("dekr_w32_no_dc", "dekr_custom"):
        return default_dekr_coco_processing_params()

    if pretrained_weights == "coco_pose" and model_name.startswith("yolo_nas_pose"):
        return default_yolo_nas_pose_coco_processing_params()

    if pretrained_weights == "imagenet" and model_name in {"vit_base", "vit_large", "vit_huge"}:
        return default_vit_imagenet_processing_params()

    if pretrained_weights == "imagenet":
        return default_imagenet_processing_params()

    if pretrained_weights == "cityscapes":
        if model_name in {"pp_lite_t_seg75", "pp_lite_b_seg75", "stdc1_seg75", "stdc2_seg75"}:
            return default_cityscapes_processing_params(0.75)
        elif model_name in {"pp_lite_t_seg50", "pp_lite_b_seg50", "stdc1_seg50", "stdc2_seg50"}:
            return default_cityscapes_processing_params(0.50)
        elif model_name in {"ddrnet_23", "ddrnet_23_slim", "ddrnet_39"}:
            return default_cityscapes_processing_params()
        elif model_name in {"segformer_b0", "segformer_b1", "segformer_b2", "segformer_b3", "segformer_b4", "segformer_b5"}:
            return default_segformer_cityscapes_processing_params()
    return dict()
