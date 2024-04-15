import torch
import torch.nn as nn
import unittest

from super_gradients.training.losses import OpticalFlowLoss
from super_gradients.training.losses.loss_utils import apply_reduce


class OpticalFlowLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_size = 100
        self.gamma = 0.8
        self.max_flow = 400
        self.reduction = "mean"
        self.batch_size = 1

    def _get_default_predictions_tensor(self, n_predictions: int, fill_value: float):
        return [torch.empty(self.batch_size, 2, self.img_size, self.img_size).fill_(fill_value) for _ in range(n_predictions)]

    def _get_default_target_tensor(self):
        return (torch.zeros(self.batch_size, 2, self.img_size, self.img_size).long(), torch.ones(self.img_size, self.img_size))

    def _assertion_flow_loss_torch_values(self, expected_value: torch.Tensor, found_value: torch.Tensor, rtol: float = 1e-5):
        self.assertTrue(torch.allclose(found_value, expected_value, rtol=rtol), msg=f"Unequal flow loss: excepted: {expected_value}, found: {found_value}")

    def test_flow_loss_l1_criterion(self):
        predictions = self._get_default_predictions_tensor(3, 2.5)
        target, valid = self._get_default_target_tensor()

        criterion = nn.L1Loss()
        loss_fn = OpticalFlowLoss(criterion=criterion, gamma=self.gamma, max_flow=self.max_flow, reduction=self.reduction)

        flow_loss = loss_fn(predictions, (target, valid))

        # expected_flow_loss
        expected_flow_loss = 0.0
        mag = torch.sum(target**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        n_predictions = len(predictions)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = i_weight * (valid[:, None] * (predictions[i] - target).abs())  # L1 dist
            expected_flow_loss += apply_reduce(i_loss, self.reduction)

        self._assertion_flow_loss_torch_values(torch.tensor(expected_flow_loss), flow_loss)

    def test_flow_loss_mse_criterion(self):
        predictions = self._get_default_predictions_tensor(3, 2.5)
        target, valid = self._get_default_target_tensor()

        criterion = nn.MSELoss()
        loss_fn = OpticalFlowLoss(criterion=criterion, gamma=self.gamma, max_flow=self.max_flow, reduction=self.reduction)

        flow_loss = loss_fn(predictions, (target, valid))

        # expected_flow_loss
        expected_flow_loss = 0.0
        mag = torch.sum(target**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < self.max_flow)

        n_predictions = len(predictions)

        for i in range(n_predictions):
            i_weight = self.gamma ** (n_predictions - i - 1)
            i_loss = i_weight * (valid[:, None] * ((predictions[i] - target) ** 2).mean())  # mse dist
            expected_flow_loss += apply_reduce(i_loss, self.reduction)

        self._assertion_flow_loss_torch_values(torch.tensor(expected_flow_loss), flow_loss)


if __name__ == "__main__":
    unittest.main()
