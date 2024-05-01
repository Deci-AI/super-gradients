from typing import Tuple

import numpy as np
from super_gradients.common.registry import register_processing
from super_gradients.training.transforms.utils import (
    _pad_image,
    PaddingCoordinates,
    _get_center_padding_coordinates,
    _rescale_bboxes,
    _get_bottom_right_padding_coordinates,
)
from super_gradients.training.utils.predict import OBBDetectionPrediction
from .processing import AutoPadding, DetectionPadToSizeMetadata, _LongestMaxSizeRescale, RescaleMetadata, _DetectionPadding


@register_processing()
class OBBDetectionCenterPadding(_DetectionPadding):
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        return _get_center_padding_coordinates(input_shape=input_shape, output_shape=self.output_shape)

    def postprocess_predictions(self, predictions: OBBDetectionPrediction, metadata: DetectionPadToSizeMetadata) -> OBBDetectionPrediction:
        offset = np.array([metadata.padding_coordinates.left, metadata.padding_coordinates.top, 0, 0, 0], dtype=np.float32).reshape(-1, 5)
        predictions.rboxes_cxcywhr = predictions.rboxes_cxcywhr - offset
        return predictions


@register_processing()
class OBBDetectionBottomRightPadding(_DetectionPadding):
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        return _get_bottom_right_padding_coordinates(input_shape=input_shape, output_shape=self.output_shape)

    def postprocess_predictions(self, predictions: OBBDetectionPrediction, metadata: DetectionPadToSizeMetadata) -> OBBDetectionPrediction:
        return predictions


@register_processing()
class OBBDetectionLongestMaxSizeRescale(_LongestMaxSizeRescale):
    def postprocess_predictions(self, predictions: OBBDetectionPrediction, metadata: RescaleMetadata) -> OBBDetectionPrediction:
        predictions.rboxes_cxcywhr = _rescale_bboxes(
            targets=predictions.rboxes_cxcywhr, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w)
        )
        return predictions


@register_processing()
class OBBDetectionAutoPadding(AutoPadding):
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPadToSizeMetadata]:
        padding_coordinates = self._get_padding_params(input_shape=image.shape[:2])  # HWC -> (H, W)
        processed_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=self.pad_value)
        return processed_image, DetectionPadToSizeMetadata(padding_coordinates=padding_coordinates)

    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        input_height, input_width = input_shape
        height_modulo, width_modulo = self.shape_multiple

        # Calculate necessary padding to reach the modulo
        padded_height = ((input_height + height_modulo - 1) // height_modulo) * height_modulo
        padded_width = ((input_width + width_modulo - 1) // width_modulo) * width_modulo

        padding_top = 0  # No padding at the top
        padding_left = 0  # No padding on the left
        padding_bottom = padded_height - input_height
        padding_right = padded_width - input_width

        return PaddingCoordinates(top=padding_top, left=padding_left, bottom=padding_bottom, right=padding_right)

    def postprocess_predictions(self, predictions: OBBDetectionPrediction, metadata: DetectionPadToSizeMetadata) -> OBBDetectionPrediction:
        offset = np.array([metadata.padding_coordinates.left, metadata.padding_coordinates.top, 0, 0, 0], dtype=np.float32).reshape(-1, 5)
        predictions.rboxes_cxcywhr = predictions.rboxes_cxcywhr + offset
        return predictions
