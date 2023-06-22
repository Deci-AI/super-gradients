import unittest

import torch

from super_gradients.training.metrics import IoU, PixelAccuracy, Dice


class TestSegmentationMetricsMultipleIgnored(unittest.TestCase):
    def test_iou_with_multiple_ignored_classes_and_absent_score(self):
        metric_multi_ignored = IoU(num_classes=5, ignore_index=[3, 1, 2])
        target_multi_ignored = torch.tensor([[3, 1, 2, 4, 4, 4]])
        pred = torch.zeros((1, 5, 6))
        pred[:, 4] = 1

        # preds after onehot -> [4,4,4,4,4,4]
        # (1 + 0)/2 : 1.0 for class 4 score and 0 for absent score for class 0
        self.assertEqual(metric_multi_ignored(pred, target_multi_ignored), 0.5)

    def test_iou_with_multiple_ignored_classes_no_absent_score(self):
        metric_multi_ignored = IoU(num_classes=5, ignore_index=[3, 1, 2])
        target_multi_ignored = torch.tensor([[3, 1, 2, 0, 4, 4]])
        pred = torch.zeros((1, 5, 6))
        pred[:, 4] = 1
        pred[0, 0, 3] = 2

        # preds after onehot -> [4,4,4,0,4,4]
        # (1 + 1)/2 : 1.0 for class 4 score and 1 for class 0
        self.assertEqual(metric_multi_ignored(pred, target_multi_ignored), 1)

    def test_dice_with_multiple_ignored_classes_and_absent_score(self):
        metric_multi_ignored = Dice(num_classes=5, ignore_index=[3, 1, 2])
        target_multi_ignored = torch.tensor([[3, 1, 2, 4, 4, 4]])
        pred = torch.zeros((1, 5, 6))
        pred[:, 4] = 1

        self.assertEqual(metric_multi_ignored(pred, target_multi_ignored), 0.5)

    def test_dice_with_multiple_ignored_classes_no_absent_score(self):
        metric_multi_ignored = Dice(num_classes=5, ignore_index=[3, 1, 2])
        target_multi_ignored = torch.tensor([[3, 1, 2, 0, 4, 4]])
        pred = torch.zeros((1, 5, 6))
        pred[:, 4] = 1
        pred[0, 0, 3] = 2

        self.assertEqual(metric_multi_ignored(pred, target_multi_ignored), 1.0)

    def test_pixelaccuracy_with_multiple_ignored_classes(self):
        metric_multi_ignored = PixelAccuracy(ignore_label=[3, 1, 2])
        target_multi_ignored = torch.tensor([[3, 1, 2, 4, 4, 4]])
        pred = torch.zeros((1, 5, 6))
        pred[:, 4] = 1

        self.assertEqual(metric_multi_ignored(pred, target_multi_ignored), 1.0)


if __name__ == "__main__":
    unittest.main()
