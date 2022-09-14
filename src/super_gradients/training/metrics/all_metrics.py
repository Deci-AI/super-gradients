from super_gradients.training.metrics import Accuracy, Top5, DetectionMetrics, IoU, PixelAccuracy, BinaryIOU, Dice,\
    BinaryDice


class MetricNames:
    """Static class holding all the supported metric names"""""
    ACCURACY = 'Accuracy'
    TOP5 = 'Top5'
    DETECTION_METRICS = 'DetectionMetrics'
    IOU = 'IoU'
    BINARY_IOU = "BinaryIOU"
    DICE = "Dice"
    BINARY_DICE = "BinaryDice"
    PIXEL_ACCURACY = 'PixelAccuracy'


METRICS = {
    MetricNames.ACCURACY: Accuracy,
    MetricNames.TOP5: Top5,
    MetricNames.DETECTION_METRICS: DetectionMetrics,
    MetricNames.IOU: IoU,
    MetricNames.BINARY_IOU: BinaryIOU,
    MetricNames.DICE: Dice,
    MetricNames.BINARY_DICE: BinaryDice,
    MetricNames.PIXEL_ACCURACY: PixelAccuracy,
}
