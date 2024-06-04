from typing import Optional, List
from functools import lru_cache

import torch
from torch import nn
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.processing_factory import ProcessingFactory
from super_gradients.module_interfaces import HasPredict
from super_gradients.training.models import CustomizableDetector
from super_gradients.training.utils.predict import ImagesDetectionPrediction
from super_gradients.training.pipelines.pipelines import SlidingWindowDetectionPipeline
from super_gradients.training.processing.processing import Processing, ComposeProcessing, DetectionAutoPadding
from super_gradients.training.utils.media.image import ImageSource
import torchvision
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback


class SlidingWindowInferenceDetectionWrapper(HasPredict, nn.Module):
    """
    Implements a sliding window inference wrapper for a customizable detector.

    :param tile_size: (int) The size of each square tile (in pixels) used in the sliding window.
    :param tile_step: (int) The step size (in pixels) between consecutive tiles in the sliding window.
    :param model: (CustomizableDetector) The detection model to which the sliding window inference is applied.
    :param min_tile_threshold: (int) Minimum dimension size for edge tiles before padding is applied.
        If the remainder of the image (after the full tiles have been applied) is smaller than this threshold,
        it will not be processed.
    :param tile_nms_iou: (Optional[float]) IoU threshold for Non-Maximum Suppression (NMS) of bounding boxes.
        Defaults to the model's internal setting if None.
    :param tile_nms_conf: (Optional[float]) Confidence threshold for predictions to consider in post-processing.
        Defaults to the model's internal setting if None.
    :param tile_nms_top_k: (Optional[int]) Maximum number of top-scoring detections to consider for NMS in each tile.
        Defaults to the model's internal setting if None.
    :param tile_nms_max_predictions: (Optional[int]) Maximum number of detections to return from each tile.
        Defaults to the model's internal setting if None.
    :param tile_nms_multi_label_per_box: (Optional[bool]) Allows multiple labels per box if True. Each anchor can produce
        multiple labels of different classes that pass the confidence threshold. Only the highest-scoring class is considered
        per anchor if False. Defaults to the model's internal setting if None.
    :param tile_nms_class_agnostic_nms: (Optional[bool]) Performs class-agnostic NMS if True, where the IoU of boxes across
        different classes is considered. Performs class-specific NMS if False. Defaults to the model's internal setting if None.
    """

    def __init__(
        self,
        tile_size: int,
        tile_step: int,
        model: Optional[CustomizableDetector],
        min_tile_threshold: int = 30,
        tile_nms_iou: Optional[float] = None,
        tile_nms_conf: Optional[float] = None,
        tile_nms_top_k: Optional[int] = None,
        tile_nms_max_predictions: Optional[int] = None,
        tile_nms_multi_label_per_box: Optional[bool] = None,
        tile_nms_class_agnostic_nms: Optional[bool] = None,
    ):

        super().__init__()
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.min_tile_threshold = min_tile_threshold

        # GENERAL DEFAULTS
        self._class_names: Optional[List[str]] = None
        self._image_processor: Optional[Processing] = None
        self._default_nms_iou: float = 0.7
        self._default_nms_conf: float = 0.5
        self._default_nms_top_k: int = 1024
        self._default_max_predictions = 300
        self._default_multi_label_per_box = True
        self._default_class_agnostic_nms = False

        # TAKE PROCESSING PARAMS FROM THE WRAPPED MODEL IF THEY ARE AVAILABLE, OTHERWISE USE THE GENERAL DEFAULTS
        self.model = model
        self.set_dataset_processing_params(**self.model.get_dataset_processing_params())

        # OVERRIDE WITH ANY EXPLICITLY PASSED PROCESSING PARAMS
        if any(
            arg is not None
            for arg in [tile_nms_iou, tile_nms_conf, tile_nms_top_k, tile_nms_max_predictions, tile_nms_multi_label_per_box, tile_nms_class_agnostic_nms]
        ):
            self.set_dataset_processing_params(
                iou=tile_nms_iou,
                conf=tile_nms_conf,
                nms_top_k=tile_nms_top_k,
                max_predictions=tile_nms_max_predictions,
                multi_label_per_box=tile_nms_multi_label_per_box,
                class_agnostic_nms=tile_nms_class_agnostic_nms,
            )
        else:

            self.sliding_window_post_prediction_callback = self.get_post_prediction_callback(
                iou=self._default_nms_iou,
                conf=self._default_nms_conf,
                nms_top_k=self._default_nms_top_k,
                max_predictions=self._default_max_predictions,
                multi_label_per_box=self._default_multi_label_per_box,
                class_agnostic_nms=self._default_class_agnostic_nms,
            )

    def forward(self, inputs: torch.Tensor, sliding_window_post_prediction_callback: Optional[DetectionPostPredictionCallback] = None) -> List[torch.Tensor]:

        sliding_window_post_prediction_callback = sliding_window_post_prediction_callback or self.sliding_window_post_prediction_callback
        batch_size, _, _, _ = inputs.shape
        all_detections = [[] for _ in range(batch_size)]  # Create a list for each image in the batch
        # Generate and process each tile
        for img_idx in range(batch_size):
            single_image = inputs[img_idx : img_idx + 1]  # Extract each image
            tiles = self._generate_tiles(single_image, self.tile_size, self.tile_step)
            for tile, (start_x, start_y) in tiles:
                tile_detections = self.model(tile)
                # Apply local NMS using post_prediction_callback
                tile_detections = sliding_window_post_prediction_callback(tile_detections)
                # Adjust detections to global image coordinates
                for img_i_tile_detections in tile_detections:
                    if len(img_i_tile_detections) > 0:
                        img_i_tile_detections[:, :4] += torch.tensor([start_x, start_y, start_x, start_y], device=tile.device)
                        all_detections[img_idx].append(img_i_tile_detections)
        # Concatenate and apply global NMS for each image's detections
        final_detections = []
        for detections in all_detections:
            if detections:
                detections = torch.cat(detections, dim=0)
                # Apply global NMS
                pred_bboxes = detections[:, :4]
                pred_cls_conf = detections[:, 4]
                pred_cls_label = detections[:, 5]
                idx_to_keep = torchvision.ops.boxes.batched_nms(
                    boxes=pred_bboxes, scores=pred_cls_conf, idxs=pred_cls_label, iou_threshold=sliding_window_post_prediction_callback.nms_threshold
                )

                final_detections.append(detections[idx_to_keep])
            else:
                final_detections.append(torch.empty(0, 6).to(inputs.device))  # Empty tensor for images with no detections
        return final_detections

    def _generate_tiles(self, image, tile_size, tile_step):
        _, _, h, w = image.shape
        tiles = []

        # Calculate the end points for the grid
        max_y = h if (h - tile_size) % tile_step < self.min_tile_threshold else h - (h - tile_size) % tile_step + tile_size
        max_x = w if (w - tile_size) % tile_step < self.min_tile_threshold else w - (w - tile_size) % tile_step + tile_size

        # Ensure that the image has enough padding if needed
        if max_y > h or max_x > w:
            padded_image = torch.zeros((image.shape[0], image.shape[1], max(max_y, h), max(max_x, w)), device=image.device)
            padded_image[:, :, :h, :w] = image  # Place the original image in the padded one
        else:
            padded_image = image

        for y in range(0, max_y - tile_size + 1, tile_step):
            for x in range(0, max_x - tile_size + 1, tile_step):
                tile = padded_image[:, :, y : y + tile_size, x : x + tile_size]
                tiles.append((tile, (x, y)))

        return tiles

    def get_post_prediction_callback(
        self, *, conf: float, iou: float, nms_top_k: int, max_predictions: int, multi_label_per_box: bool, class_agnostic_nms: bool
    ) -> DetectionPostPredictionCallback:
        """
        Get a post prediction callback for this model.

        :param conf:                A minimum confidence threshold for predictions to be used in post-processing.
        :param iou:                 A IoU threshold for boxes non-maximum suppression.
        :param nms_top_k:           The maximum number of detections to consider for the NMS applied on each tile.
        :param max_predictions:     The maximum number of detections to return in each tile.
        :param multi_label_per_box: If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :return:
        """
        return self.model.get_post_prediction_callback(
            conf=conf,
            iou=iou,
            nms_top_k=nms_top_k,
            max_predictions=max_predictions,
            multi_label_per_box=multi_label_per_box,
            class_agnostic_nms=class_agnostic_nms,
        )

    @resolve_param("image_processor", ProcessingFactory())
    def set_dataset_processing_params(
        self,
        class_names: Optional[List[str]] = None,
        image_processor: Optional[Processing] = None,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
    ) -> None:
        """Set the processing parameters for the dataset.

        :param class_names:         (Optional) Names of the dataset the model was trained on.
        :param image_processor:     (Optional) Image processing objects to reproduce the dataset preprocessing used for training.
        :param iou:                 (Optional) IoU threshold for the nms algorithm applied.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS in each tile.
        :param max_predictions:     (Optional) The maximum number of detections to return in each tile.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        """
        if class_names is not None:
            self._class_names = tuple(class_names)
        if image_processor is not None:
            self._image_processor = image_processor

        if iou is None:
            iou = self._default_nms_iou
        if conf is None:
            conf = self._default_nms_conf
        if nms_top_k is None:
            nms_top_k = self._default_nms_top_k
        if max_predictions is None:
            max_predictions = self._default_max_predictions
        if multi_label_per_box is None:
            multi_label_per_box = self._default_multi_label_per_box
        if class_agnostic_nms is None:
            class_agnostic_nms = self._default_class_agnostic_nms

        self.sliding_window_post_prediction_callback = self.get_post_prediction_callback(
            iou=float(iou),
            conf=float(conf),
            nms_top_k=int(nms_top_k),
            max_predictions=int(max_predictions),
            multi_label_per_box=bool(multi_label_per_box),
            class_agnostic_nms=bool(class_agnostic_nms),
        )

    def get_processing_params(self) -> Optional[Processing]:
        return self._image_processor

    @lru_cache(maxsize=1)
    def _get_pipeline(
        self,
        *,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
        fp16: bool = True,
    ) -> SlidingWindowDetectionPipeline:
        """Instantiate the prediction pipeline of this model.

        :param iou:                 (Optional) IoU threshold for the nms algorithm.
         If None, the default value associated to the training is used.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded.
                                    If None, the default value associated to the training is used.
        :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS for each tile.
        :param max_predictions:     (Optional) The maximum number of detections to return for each tile.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :param fp16:                If True, use mixed precision for inference.
        """
        if None in (self._class_names, self._image_processor, self._default_nms_iou, self._default_nms_conf):
            raise RuntimeError(
                "You must set the dataset processing parameters before calling predict.\n"
                "Please call "
                "`model.set_dataset_processing_params(...)` first or do so on self.model. "
            )

        iou = self._default_nms_iou if iou is None else iou
        conf = self._default_nms_conf if conf is None else conf
        nms_top_k = self._default_nms_top_k if nms_top_k is None else nms_top_k
        max_predictions = self._default_max_predictions if max_predictions is None else max_predictions
        multi_label_per_box = self._default_multi_label_per_box if multi_label_per_box is None else multi_label_per_box
        class_agnostic_nms = self._default_class_agnostic_nms if class_agnostic_nms is None else class_agnostic_nms

        # Ensure that the image size is divisible by 32.
        if isinstance(self._image_processor, ComposeProcessing) and skip_image_resizing:
            image_processor = self._image_processor.get_equivalent_compose_without_resizing(
                auto_padding=DetectionAutoPadding(shape_multiple=(32, 32), pad_value=0)
            )
        else:
            image_processor = self._image_processor

        pipeline = SlidingWindowDetectionPipeline(
            model=self,
            image_processor=image_processor,
            post_prediction_callback=self.get_post_prediction_callback(
                iou=iou,
                conf=conf,
                nms_top_k=nms_top_k,
                max_predictions=max_predictions,
                multi_label_per_box=multi_label_per_box,
                class_agnostic_nms=class_agnostic_nms,
            ),
            class_names=self._class_names,
            fuse_model=fuse_model,
            fp16=fp16,
        )
        return pipeline

    def predict(
        self,
        images: ImageSource,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        batch_size: int = 32,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
        fp16: bool = True,
    ) -> ImagesDetectionPrediction:
        """Predict an image or a list of images.

        :param images:              Images to predict.
        :param iou:                 (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded.
                                    If None, the default value associated to the training is used.
        :param batch_size:          Maximum number of images to process at the same time.
        :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS.
        :param max_predictions:     (Optional) The maximum number of detections to return.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :param fp16:                        If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(
            iou=iou,
            conf=conf,
            fuse_model=fuse_model,
            skip_image_resizing=skip_image_resizing,
            nms_top_k=nms_top_k,
            max_predictions=max_predictions,
            multi_label_per_box=multi_label_per_box,
            class_agnostic_nms=class_agnostic_nms,
            fp16=fp16,
        )
        return pipeline(images, batch_size=batch_size)  # type: ignore

    def predict_webcam(
        self,
        iou: Optional[float] = None,
        conf: Optional[float] = None,
        fuse_model: bool = True,
        skip_image_resizing: bool = False,
        nms_top_k: Optional[int] = None,
        max_predictions: Optional[int] = None,
        multi_label_per_box: Optional[bool] = None,
        class_agnostic_nms: Optional[bool] = None,
        fp16: bool = True,
    ):
        """Predict using webcam.

        :param iou:                 (Optional) IoU threshold for the nms algorithm. If None, the default value associated to the training is used.
        :param conf:                (Optional) Below the confidence threshold, prediction are discarded.
                                    If None, the default value associated to the training is used.
        :param fuse_model:          If True, create a copy of the model, and fuse some of its layers to increase performance. This increases memory usage.
        :param skip_image_resizing: If True, the image processor will not resize the images.
        :param nms_top_k:           (Optional) The maximum number of detections to consider for NMS.
        :param max_predictions:     (Optional) The maximum number of detections to return.
        :param multi_label_per_box: (Optional) If True, each anchor can produce multiple labels of different classes.
                                    If False, each anchor can produce only one label of the class with the highest score.
        :param class_agnostic_nms:  (Optional) If True, perform class-agnostic NMS (i.e IoU of boxes of different classes is checked).
                                    If False NMS is performed separately for each class.
        :param fp16:                If True, use mixed precision for inference.
        """
        pipeline = self._get_pipeline(
            iou=iou,
            conf=conf,
            fuse_model=fuse_model,
            skip_image_resizing=skip_image_resizing,
            nms_top_k=nms_top_k,
            max_predictions=max_predictions,
            multi_label_per_box=multi_label_per_box,
            class_agnostic_nms=class_agnostic_nms,
            fp16=fp16,
        )
        pipeline.predict_webcam()

    def get_input_channels(self) -> int:
        return self.model.get_input_channels()
