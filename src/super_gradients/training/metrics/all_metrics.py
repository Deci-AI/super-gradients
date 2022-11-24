from super_gradients.common.object_names import Metrics
from super_gradients.training.metrics import (
    Accuracy,
    Top5,
    DetectionMetrics,
    IoU,
    PixelAccuracy,
    BinaryIOU,
    Dice,
    BinaryDice,
    DetectionMetrics_050,
    DetectionMetrics_075,
    DetectionMetrics_050_095,
)


METRICS = {
    Metrics.ACCURACY: Accuracy,
    Metrics.TOP5: Top5,
    Metrics.DETECTION_METRICS: DetectionMetrics,
    Metrics.DETECTION_METRICS_050: DetectionMetrics_050,
    Metrics.DETECTION_METRICS_075: DetectionMetrics_075,
    Metrics.DETECTION_METRICS_050_095: DetectionMetrics_050_095,
    Metrics.IOU: IoU,
    Metrics.BINARY_IOU: BinaryIOU,
    Metrics.DICE: Dice,
    Metrics.BINARY_DICE: BinaryDice,
    Metrics.PIXEL_ACCURACY: PixelAccuracy,
}
