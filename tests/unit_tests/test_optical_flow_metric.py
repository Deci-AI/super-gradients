import torch
import unittest

from super_gradients.training.metrics.optical_flow_metric import EPE


class TestOpticalFlowMetric(unittest.TestCase):
    def test_epe_metric(self):
        # Specific example data
        pred_flow = [torch.ones(1, 2, 100, 100)]
        gt_flow = torch.zeros(1, 2, 100, 100)
        valid = torch.ones(100, 100)

        # Create instances of delta metrics
        max_flow = 400
        metric = EPE(max_flow=max_flow)

        # Update metrics with specific example data
        metric.update(pred_flow, (gt_flow, valid))

        # Expected metric
        mag = torch.sum(gt_flow**2, dim=1).sqrt()
        valid = (valid >= 0.5) & (mag < max_flow)

        expected_epe = torch.sum((pred_flow[-1] - gt_flow) ** 2, dim=1).sqrt()
        expected_epe = expected_epe.view(-1)[valid.view(-1)]
        expected_epe = expected_epe.mean().item()

        # Compute and assert the delta metrics
        self.assertAlmostEqual(metric.compute()["epe"], expected_epe)


if __name__ == "__main__":
    unittest.main()
