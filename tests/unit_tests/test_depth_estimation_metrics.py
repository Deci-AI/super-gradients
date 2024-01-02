import torch
import unittest

from super_gradients.training.metrics import Delta1, Delta2, Delta3, DepthMAE, DepthMAPE, DepthMSE, DepthRMSE, DepthMSLE


class TestDepthEstimationMetrics(unittest.TestCase):
    def test_delta_metrics(self):
        # Specific example data
        pred_depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        gt_depth = torch.tensor([[[1.5, 2.5], [3.5, 4.5]]], dtype=torch.float32)

        # Create instances of delta metrics
        delta1_metric = Delta1()
        delta2_metric = Delta2()
        delta3_metric = Delta3()

        # Update metrics with specific example data
        delta1_metric.update(pred_depth, gt_depth)
        delta2_metric.update(pred_depth, gt_depth)
        delta3_metric.update(pred_depth, gt_depth)

        # Compute and assert the delta metrics
        self.assertAlmostEqual(delta1_metric.compute().item(), 0.5)
        self.assertAlmostEqual(delta2_metric.compute().item(), 1.0)
        self.assertAlmostEqual(delta3_metric.compute().item(), 1.0)

    def test_mae_metric(self):
        # Specific example data
        pred_depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        gt_depth = torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=torch.float32)

        # Create instances of MAE and MAPE metrics
        mae_metric = DepthMAE(ignore_val=-1)

        # Update metrics with specific example data
        mae_metric.update(pred_depth, gt_depth)

        # Compute and assert the MAE metric
        self.assertAlmostEqual(mae_metric.compute().item(), 0.5, places=5)

    def test_mape_metric(self):
        # Specific example data
        pred_depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        gt_depth = torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=torch.float32)

        # Create an instance of MAPE metric
        mape_metric = DepthMAPE()

        # Update metric with specific example data
        mape_metric.update(pred_depth, gt_depth)

        # Compute and assert the MAPE metric
        self.assertAlmostEqual(mape_metric.compute().item(), (0.5 / 1.5 + 0.5 / 2.5 + 0.5 / 3.5 + 0.5 / 4.5) / 4)

    def test_mse_metric(self):
        # Specific example data
        pred_depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        gt_depth = torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=torch.float32)

        # Create an instance of MSE metric
        mse_metric = DepthMSE()

        # Update metric with specific example data
        mse_metric.update(pred_depth, gt_depth)

        # Compute and assert the MSE metric
        self.assertAlmostEqual(mse_metric.compute().item(), 0.25)

    def test_rmse_metric(self):
        # Specific example data
        pred_depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        gt_depth = torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=torch.float32)

        # Create an instance of RMSE metric
        rmse_metric = DepthRMSE()

        # Update metric with specific example data
        rmse_metric.update(pred_depth, gt_depth)

        # Compute and assert the RMSE metric
        self.assertAlmostEqual(rmse_metric.compute().item(), 0.5)

    def test_msle_metric(self):
        # Specific example data
        pred_depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        gt_depth = torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=torch.float32)

        # Create an instance of MSLE metric
        msle_metric = DepthMSLE()

        # Update metric with specific example data
        msle_metric.update(pred_depth, gt_depth)

        # Compute and assert the MSLE metric
        self.assertAlmostEqual(msle_metric.compute().item(), 0.024128085002303123)


if __name__ == "__main__":
    unittest.main()
