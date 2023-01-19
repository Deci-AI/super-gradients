import torch
import unittest

from super_gradients.training.losses.ddrnet_loss import DDRNetLoss


class DDRLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.num_classes = 19
        self.img_size = 256
        self.eps = 0.01

        # equal probability distribution, p = 1 / num_classes
        # except loss to be: -log(p)
        self.predictions = torch.ones((1, self.num_classes, self.img_size, self.img_size))
        self.targets = torch.randint(0, self.num_classes, (1, self.img_size, self.img_size))

    def test_without_auxiliary_loss(self):
        """
        No Auxiliary loss, only one prediction map
        """
        weights = [1.0]
        criterion = DDRNetLoss((1 / self.num_classes - self.eps), ohem_percentage=0.1, weights=weights)
        bce_loss = -torch.log(torch.tensor(1 / self.num_classes))
        expected_loss = bce_loss * weights[0]
        loss, _ = criterion(self.predictions, self.targets)

        self.assertAlmostEqual(expected_loss, loss, delta=1e-5)

    def test_with_auxiliary_loss(self):
        """
        Auxiliary loss, 2 prediction maps, as DDRNet paper.
        """
        predictions = [self.predictions, self.predictions]
        weights = [1.0, 0.4]

        criterion = DDRNetLoss((1 / self.num_classes - self.eps), ohem_percentage=0.1, weights=weights)
        expected_loss = -torch.log(torch.tensor(1 / self.num_classes)) * weights[0] + -torch.log(torch.tensor(1 / self.num_classes)) * weights[1]
        loss, _ = criterion(predictions, self.targets)

        self.assertAlmostEqual(expected_loss, loss, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
