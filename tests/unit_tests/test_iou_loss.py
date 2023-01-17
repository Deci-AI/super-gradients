import torch
import unittest

from super_gradients.training.losses.iou_loss import IoULoss, GeneralizedIoULoss, BinaryIoULoss


class IoULossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_size = 32
        self.eps = 1e-5
        self.num_classes = 2

    def _get_default_predictions_tensor(self, fill_value: float):
        return torch.empty(3, self.num_classes, self.img_size, self.img_size).fill_(fill_value)

    def _get_default_target_zeroes_tensor(self):
        return torch.zeros((3, self.img_size, self.img_size)).long()

    def _assertion_iou_torch_values(self, expected_value: torch.Tensor, found_value: torch.Tensor, rtol: float = 1e-5):
        self.assertTrue(torch.allclose(found_value, expected_value, rtol=rtol), msg=f"Unequal iou loss: excepted: {expected_value}, found: {found_value}")

    def test_iou(self):
        predictions = self._get_default_predictions_tensor(0.0)
        # only label 0 is predicted as positive.
        predictions[:, 0] = 1.0
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, the other half with 1.
        target[:, : self.img_size // 2] = 1

        intersection = torch.tensor([0.5, 0.0])
        union = torch.tensor([1.0, 0.5])

        expected_iou_loss = 1.0 - (intersection / (union + self.eps))
        expected_iou_loss = expected_iou_loss.mean()

        criterion = IoULoss(smooth=0, eps=self.eps, apply_softmax=False)
        iou_loss = criterion(predictions, target)

        self._assertion_iou_torch_values(expected_iou_loss, iou_loss)

    def test_iou_binary(self):
        # all predictions are 0.6
        predictions = torch.ones((1, 1, self.img_size, self.img_size)) * 0.6
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, the other half with 1.
        target[:, : self.img_size // 2] = 1

        intersection = torch.tensor([0.6 * 0.5])
        union = torch.tensor([0.6 + 0.5 - 0.6 * 0.5])

        expected_iou_loss = 1.0 - (intersection / (union + self.eps))
        expected_iou_loss = expected_iou_loss.mean()

        criterion = BinaryIoULoss(smooth=0, eps=self.eps, apply_sigmoid=False)
        iou_loss = criterion(predictions, target)

        self._assertion_iou_torch_values(expected_iou_loss, iou_loss, rtol=1e-3)

    def test_iou_weight_classes(self):
        weight = torch.tensor([0.25, 0.66])
        predictions = self._get_default_predictions_tensor(0.0)
        # only label 0 is predicted as positive.
        predictions[:, 0] = 1.0
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, the other half with 1.
        target[:, : self.img_size // 2] = 1

        intersection = torch.tensor([0.5, 0.0])
        union = torch.tensor([1.0, 0.5])

        expected_iou_loss = 1.0 - (intersection / (union + self.eps))
        expected_iou_loss *= weight
        expected_iou_loss = expected_iou_loss.mean()

        criterion = IoULoss(smooth=0, eps=self.eps, apply_softmax=False, weight=weight)
        iou_loss = criterion(predictions, target)

        self._assertion_iou_torch_values(expected_iou_loss, iou_loss)

    def test_iou_with_ignore(self):
        ignore_index = 2
        predictions = self._get_default_predictions_tensor(0.0)
        # only label 0 is predicted as positive.
        predictions[:, 0] = 1.0
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, quarter with 1 and quarter with ignore.
        target[:, : self.img_size // 2, : self.img_size // 2] = 1
        target[:, : self.img_size // 2, self.img_size // 2 :] = ignore_index
        # ignore samples are excluded in both intersection and union.
        intersection = torch.tensor([0.5, 0.0])
        union = torch.tensor([0.75, 0.25])

        expected_iou_loss = 1.0 - (intersection / (union + self.eps))
        expected_iou_loss = expected_iou_loss.mean()

        criterion = IoULoss(smooth=0, eps=self.eps, apply_softmax=False, ignore_index=ignore_index)
        iou_loss = criterion(predictions, target)

        self._assertion_iou_torch_values(expected_iou_loss, iou_loss)

    def test_generalized_iou(self):
        predictions = self._get_default_predictions_tensor(0.0)
        # half prediction are 0 class, the other half 1 class.
        predictions[:, 0, : self.img_size // 2] = 1.0
        predictions[:, 1, self.img_size // 2 :] = 1.0
        # only 0 class in target.
        target = self._get_default_target_zeroes_tensor()

        intersection = torch.tensor([0.5, 0.0])
        union = torch.tensor([1.0, 0.5])
        counts = torch.tensor([target.numel(), 0.0])
        weights = 1 / (counts**2)
        weights[1] = 0.0  # instead of inf

        eps = 1e-17
        expected_iou_loss = 1.0 - ((weights * intersection) / (weights * union + eps))
        expected_iou_loss = expected_iou_loss.mean()

        criterion = GeneralizedIoULoss(smooth=0, eps=eps, apply_softmax=False)
        iou_loss = criterion(predictions, target)

        self._assertion_iou_torch_values(expected_iou_loss, iou_loss)


if __name__ == "__main__":
    unittest.main()
