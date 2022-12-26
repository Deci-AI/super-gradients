import torch
import unittest

from super_gradients.training.losses.dice_loss import DiceLoss, GeneralizedDiceLoss, BinaryDiceLoss


class DiceLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_size = 32
        self.eps = 1e-5
        self.num_classes = 2

    def _get_default_predictions_tensor(self, fill_value: float):
        return torch.empty(3, self.num_classes, self.img_size, self.img_size).fill_(fill_value)

    def _get_default_target_zeroes_tensor(self):
        return torch.zeros((3, self.img_size, self.img_size)).long()

    def _assertion_dice_torch_values(self, expected_value: torch.Tensor, found_value: torch.Tensor, rtol: float = 1e-5):
        self.assertTrue(torch.allclose(found_value, expected_value, rtol=rtol), msg=f"Unequal dice loss: excepted: {expected_value}, found: {found_value}")

    def test_dice(self):
        predictions = self._get_default_predictions_tensor(0.0)
        # only label 0 is predicted as positive.
        predictions[:, 0] = 1.0
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, the other half with 1.
        target[:, : self.img_size // 2] = 1

        intersection = torch.tensor([0.5, 0.0])
        denominator = torch.tensor([1.5, 0.5])

        expected_dice_loss = 1.0 - ((2.0 * intersection) / (denominator + self.eps))
        expected_dice_loss = expected_dice_loss.mean()

        criterion = DiceLoss(smooth=0, eps=self.eps, apply_softmax=False)
        dice_loss = criterion(predictions, target)

        self._assertion_dice_torch_values(expected_dice_loss, dice_loss)

    def test_dice_binary(self):
        # all predictions are 0.6
        predictions = torch.ones((1, 1, self.img_size, self.img_size)) * 0.6
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, the other half with 1.
        target[:, : self.img_size // 2] = 1

        intersection = torch.tensor([0.6 * 0.5])
        denominator = torch.tensor([0.6 + 0.5])

        expected_dice_loss = 1.0 - ((2.0 * intersection) / (denominator + self.eps))
        expected_dice_loss = expected_dice_loss.mean()

        criterion = BinaryDiceLoss(smooth=0, eps=self.eps, apply_sigmoid=False)
        dice_loss = criterion(predictions, target)

        self._assertion_dice_torch_values(expected_dice_loss, dice_loss, rtol=1e-3)

    def test_dice_weight_classes(self):
        weight = torch.tensor([0.25, 0.66])
        predictions = self._get_default_predictions_tensor(0.0)
        # only label 0 is predicted as positive.
        predictions[:, 0] = 1.0
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, the other half with 1.
        target[:, : self.img_size // 2] = 1

        intersection = torch.tensor([0.5, 0.0])
        denominator = torch.tensor([1.5, 0.5])

        expected_dice_loss = 1.0 - ((2.0 * intersection) / (denominator + self.eps))
        expected_dice_loss *= weight
        expected_dice_loss = expected_dice_loss.mean()

        criterion = DiceLoss(smooth=0, eps=self.eps, apply_softmax=False, weight=weight)
        dice_loss = criterion(predictions, target)

        self._assertion_dice_torch_values(expected_dice_loss, dice_loss)

    def test_dice_with_ignore(self):
        ignore_index = 2
        predictions = self._get_default_predictions_tensor(0.0)
        # only label 0 is predicted as positive.
        predictions[:, 0] = 1.0
        target = self._get_default_target_zeroes_tensor()
        # half target with label 0, quarter with 1 and quarter with ignore.
        target[:, : self.img_size // 2, : self.img_size // 2] = 1
        target[:, : self.img_size // 2, self.img_size // 2 :] = ignore_index
        # ignore samples are excluded in both intersection and denominator.
        intersection = torch.tensor([0.5, 0.0])
        denominator = torch.tensor([0.75 + 0.5, 0.25])

        expected_dice_loss = 1.0 - ((2.0 * intersection) / (denominator + self.eps))
        expected_dice_loss = expected_dice_loss.mean()

        criterion = DiceLoss(smooth=0, eps=self.eps, apply_softmax=False, ignore_index=ignore_index)
        dice_loss = criterion(predictions, target)

        self._assertion_dice_torch_values(expected_dice_loss, dice_loss)

    def test_generalized_dice(self):
        predictions = self._get_default_predictions_tensor(0.0)
        # half prediction are 0 class, the other half 1 class.
        predictions[:, 0, : self.img_size // 2] = 1.0
        predictions[:, 1, self.img_size // 2 :] = 1.0
        # only 0 class in target.
        target = self._get_default_target_zeroes_tensor()

        intersection = torch.tensor([0.5, 0.0])
        denominator = torch.tensor([1.5, 0.5])
        counts = torch.tensor([target.numel(), 0.0])
        weights = 1 / (counts**2)
        weights[1] = 0.0  # instead of inf

        eps = 1e-17
        expected_dice_loss = 1.0 - ((2.0 * weights * intersection) / (weights * denominator + eps))
        expected_dice_loss = expected_dice_loss.mean()

        criterion = GeneralizedDiceLoss(smooth=0, eps=eps, apply_softmax=False)
        dice_loss = criterion(predictions, target)

        self._assertion_dice_torch_values(expected_dice_loss, dice_loss)


if __name__ == "__main__":
    unittest.main()
