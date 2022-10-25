import torch
import unittest
import torch.nn.functional as F

from super_gradients.training.losses.ohem_ce_loss import OhemCELoss


class OhemLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.img_size = 64
        self.eps = 0.01

    def test_all_hard_no_mining(self):
        # equal probability distribution, p = 1 / num_classes
        # except loss to be: -log(p)
        num_classes = 19
        targets = torch.randint(0, num_classes, (1, self.img_size, self.img_size))
        predictions = torch.ones((1, num_classes, self.img_size, self.img_size))

        probability = 1 / num_classes

        # All samples are hard, No Hard-mining
        criterion = OhemCELoss(threshold=probability + self.eps, mining_percent=0.1)
        expected_loss = -torch.log(torch.tensor(probability))
        loss = criterion(predictions, targets)

        self.assertAlmostEqual(expected_loss, loss, delta=1e-5)

    def test_hard_mining(self):
        num_classes = 2
        predictions = torch.ones((1, num_classes, self.img_size, self.img_size))
        targets = torch.randint(0, num_classes, (1, self.img_size, self.img_size))

        # create hard samples
        hard_class = 0
        mask = targets == hard_class
        predictions[:, hard_class, mask.squeeze()] = 0.0

        hard_percent = mask.sum() / targets.numel()

        predicted_prob = F.softmax(torch.tensor([0.0, 1.0]), dim=0)[0].item()

        criterion = OhemCELoss(threshold=predicted_prob + self.eps, mining_percent=hard_percent)
        expected_loss = -torch.log(torch.tensor(predicted_prob))
        loss = criterion(predictions, targets)

        self.assertAlmostEqual(expected_loss, loss, delta=1e-5)

    def test_ignore_label(self):
        num_classes = 2
        predictions = torch.ones((1, num_classes, self.img_size, self.img_size))
        targets = torch.randint(0, num_classes, (1, self.img_size, self.img_size))

        # create hard samples, to be ignored later
        hard_class = 0
        mask = targets == hard_class
        predictions[:, hard_class, mask.squeeze()] = 0.0

        # except loss to be an equal distribution, w.r.t ignoring the hard label
        predicted_prob = F.softmax(torch.tensor([1.0, 1.0]), dim=0)[0].item()

        criterion = OhemCELoss(threshold=predicted_prob + self.eps, mining_percent=1.0, ignore_lb=hard_class)
        expected_loss = -torch.log(torch.tensor(predicted_prob))
        loss = criterion(predictions, targets)

        self.assertAlmostEqual(expected_loss, loss, delta=1e-5)

    def test_all_are_ignore_label(self):
        num_classes = 2
        predictions = torch.ones((1, num_classes, self.img_size, self.img_size))
        targets = torch.zeros(1, self.img_size, self.img_size).long()  # all targets are 0 class
        ignore_class = 0

        criterion = OhemCELoss(threshold=0.5, mining_percent=1.0, ignore_lb=ignore_class)
        expected_loss = 0.0  # except empty zero tensor, because all are ignore labels
        loss = criterion(predictions, targets)

        self.assertAlmostEqual(expected_loss, loss, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
