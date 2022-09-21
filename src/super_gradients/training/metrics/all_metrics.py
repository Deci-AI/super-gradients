from super_gradients.training.object_names import Metrics
from super_gradients.training.metrics import Accuracy, Top5, DetectionMetrics, IoU, PixelAccuracy, BinaryIOU, Dice,\
    BinaryDice


METRICS = {
    Metrics.ACCURACY: Accuracy,
    Metrics.TOP5: Top5,
    Metrics.DETECTION_METRICS: DetectionMetrics,
    Metrics.IOU: IoU,
    Metrics.BINARY_IOU: BinaryIOU,
    Metrics.DICE: Dice,
    Metrics.BINARY_DICE: BinaryDice,
    Metrics.PIXEL_ACCURACY: PixelAccuracy,
}
