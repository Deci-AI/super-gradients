import torch
import unittest
import torch.nn as nn
from super_gradients.training.losses.mask_loss import MaskAttentionLoss
from super_gradients.training.utils.segmentation_utils import to_one_hot


class MaskAttentionLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_size = 32
        self.num_classes = 4
        self.batch = 3
        torch.manual_seed(65)

    def _get_default_predictions_tensor(self):
        return torch.randn(self.batch, self.num_classes, self.img_size, self.img_size)

    def _get_default_target_tensor(self):
        return torch.randint(0, self.num_classes, size=(self.batch, self.img_size, self.img_size))

    def _get_default_mask_tensor(self):
        mask = torch.zeros(self.batch, 1, self.img_size, self.img_size)
        # half tensor rows as 1
        mask[:, :, self.img_size // 2 :] = 1
        return mask.float()

    def _assertion_torch_values(self, expected_value: torch.Tensor, found_value: torch.Tensor, rtol: float = 1e-5):
        self.assertTrue(torch.allclose(found_value, expected_value, rtol=rtol), msg=f"Unequal torch tensors: excepted: {expected_value}, found: {found_value}")

    def test_with_cross_entropy_loss(self):
        """
        Test simple case using CrossEntropyLoss,
        shapes: predict [BxCxHxW], target [BxHxW], mask [Bx1xHxW]
        """
        predict = torch.randn(self.batch, self.num_classes, self.img_size, self.img_size)
        target = self._get_default_target_tensor()
        mask = self._get_default_mask_tensor()

        loss_weigths = [1.0, 0.5]
        ce_crit = nn.CrossEntropyLoss(reduction="none")
        mask_ce_crit = MaskAttentionLoss(criterion=ce_crit, loss_weights=loss_weigths)

        # expected result
        ce_loss = ce_crit(predict, target)
        _mask = mask.view_as(ce_loss)
        mask_loss = ce_loss * _mask
        mask_loss = mask_loss[_mask == 1]  # consider only mask samples for mask loss computing
        expected_loss = ce_loss.mean() * loss_weigths[0] + mask_loss.mean() * loss_weigths[1]

        # mask ce loss result
        loss = mask_ce_crit(predict, target, mask)

        self._assertion_torch_values(expected_loss, loss)

    def test_with_binary_cross_entropy_loss(self):
        """
        Test case using BCEWithLogitsLoss, where mask is a spatial mask applied across all channels.
        shapes: predict [BxCxHxW], target (one-hot) [BxCxHxW], mask [Bx1xHxW]
        """
        predict = self._get_default_predictions_tensor()
        target = torch.randn(self.batch, self.num_classes, self.img_size, self.img_size)
        mask = self._get_default_mask_tensor()

        loss_weigths = [1.0, 0.5]
        ce_crit = nn.BCEWithLogitsLoss(reduction="none")
        mask_ce_crit = MaskAttentionLoss(criterion=ce_crit, loss_weights=loss_weigths)

        # expected result
        ce_loss = ce_crit(predict, target)
        _mask = mask.expand_as(ce_loss)
        mask_loss = ce_loss * _mask
        mask_loss = mask_loss[_mask == 1]  # consider only mask samples for mask loss computing
        expected_loss = ce_loss.mean() * loss_weigths[0] + mask_loss.mean() * loss_weigths[1]

        # mask ce loss result
        loss = mask_ce_crit(predict, target, mask)

        self._assertion_torch_values(expected_loss, loss)

    def test_reduction_none(self):
        """
        Test case mask loss with reduction="none".
        shapes: predict [BxCxHxW], target [BxHxW], mask [Bx1xHxW], except output to be same as target shape.
        """
        predict = torch.randn(self.batch, self.num_classes, self.img_size, self.img_size)
        target = self._get_default_target_tensor()
        mask = self._get_default_mask_tensor()

        loss_weigths = [1.0, 0.5]
        ce_crit = nn.CrossEntropyLoss(reduction="none")
        mask_ce_crit = MaskAttentionLoss(criterion=ce_crit, loss_weights=loss_weigths, reduction="none")

        # expected result
        ce_loss = ce_crit(predict, target)
        _mask = mask.view_as(ce_loss)
        mask_loss = ce_loss * _mask
        expected_loss = ce_loss * loss_weigths[0] + mask_loss * loss_weigths[1]

        # mask ce loss result
        loss = mask_ce_crit(predict, target, mask)

        self._assertion_torch_values(expected_loss, loss)
        self.assertEqual(target.size(), loss.size())

    def test_assert_valid_arguments(self):
        # ce_criterion reduction must be none
        kwargs = {"criterion": nn.CrossEntropyLoss(reduction="mean")}
        self.failUnlessRaises(ValueError, MaskAttentionLoss, **kwargs)
        # loss_weights must have only 2 values
        kwargs = {"criterion": nn.CrossEntropyLoss(reduction="none"), "loss_weights": [1.0, 1.0, 1.0]}
        self.failUnlessRaises(ValueError, MaskAttentionLoss, **kwargs)
        # mask loss_weight must be a positive value
        kwargs = {"criterion": nn.CrossEntropyLoss(reduction="none"), "loss_weights": [1.0, 0.0]}
        self.failUnlessRaises(ValueError, MaskAttentionLoss, **kwargs)

    def test_multi_class_mask(self):
        """
        Test case using MSELoss, where there is different spatial masks per channel.
        shapes: predict [BxCxHxW], target [BxCxHxW], mask [BxCxHxW]
        """
        predict = self._get_default_predictions_tensor()
        # when using bce loss, target is usually a one hot vector and must be with the same shape as the prediction.
        target = self._get_default_target_tensor()
        target = to_one_hot(target, self.num_classes).float()
        mask = torch.randint(0, 2, size=(self.batch, self.num_classes, self.img_size, self.img_size)).float()

        loss_weigths = [1.0, 0.5]
        ce_crit = nn.MSELoss(reduction="none")
        mask_ce_crit = MaskAttentionLoss(criterion=ce_crit, loss_weights=loss_weigths)

        # expected result
        mse_loss = ce_crit(predict, target)
        mask_loss = mse_loss * mask
        mask_loss = mask_loss[mask == 1]  # consider only mask samples for mask loss computing
        expected_loss = mse_loss.mean() * loss_weigths[0] + mask_loss.mean() * loss_weigths[1]

        # mask ce loss result
        loss = mask_ce_crit(predict, target, mask)

        self._assertion_torch_values(expected_loss, loss)

    def test_broadcast_exceptions(self):
        """
        Test assertion in mask broadcasting
        """
        predict = torch.randn(self.batch, self.num_classes, self.img_size, self.img_size)
        target = torch.randint(0, self.num_classes, size=(self.batch, self.num_classes, self.img_size, self.img_size)).float()

        loss_weigths = [1.0, 0.5]
        ce_crit = nn.BCEWithLogitsLoss(reduction="none")
        mask_ce_crit = MaskAttentionLoss(criterion=ce_crit, loss_weights=loss_weigths)

        # mask with wrong spatial size.
        mask = torch.zeros(self.batch, self.img_size, 1).float()
        self.failUnlessRaises(AssertionError, mask_ce_crit, *(predict, target, mask))

        # mask with wrong batch size.
        mask = torch.zeros(self.batch + 1, self.img_size, self.img_size).float()
        self.failUnlessRaises(AssertionError, mask_ce_crit, *(predict, target, mask))

        # mask with invalid channels num.
        mask = torch.zeros(self.batch, 2, self.img_size, self.img_size).float()
        self.failUnlessRaises(AssertionError, mask_ce_crit, *(predict, target, mask))

    def test_with_cross_entropy_loss_maskless(self):
        """
        Test case with mask filled with zeros, corresponding to a scenario without
        attention. It's expected that the mask doesn't contribute to the loss.

        This scenario may happen when using edge masks on an image without
        edges - there's only one semantic region in the whole image.

        Shapes: predict [BxCxHxW], target [BxHxW], mask [Bx1xHxW]
        """
        predict = torch.randn(self.batch, self.num_classes, self.img_size, self.img_size)
        target = self._get_default_target_tensor()
        # Create a mask filled with zeros to disable the attention component
        mask = self._get_default_mask_tensor() * 0.0

        loss_weigths = [1.0, 0.5]
        ce_crit = nn.CrossEntropyLoss(reduction="none")
        mask_ce_crit = MaskAttentionLoss(criterion=ce_crit, loss_weights=loss_weigths)

        # expected result - no contribution from mask
        ce_loss = ce_crit(predict, target)
        expected_loss = ce_loss.mean() * loss_weigths[0]

        # mask ce loss result
        loss = mask_ce_crit(predict, target, mask)

        self._assertion_torch_values(expected_loss, loss)


if __name__ == "__main__":
    unittest.main()
