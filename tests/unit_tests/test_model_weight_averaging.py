import collections
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from super_gradients.training.utils.weight_averaging_utils import ModelWeightAveraging


class TestModelWeightAveraging(unittest.TestCase):
    def test_model_weight_averaging_single_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            avg = ModelWeightAveraging(
                ckpt_dir=tmp_dir,
                greater_is_better=True,
                metric_to_watch="acc",
                load_checkpoint=False,
                number_of_models_to_average=10,
            )

            model = self._create_dummy_model()
            model_sd = model.state_dict()
            avg_model_sd = avg.get_average_model(model, {"acc": 0.99})
            self.assertStateDictAlmostEqual(avg_model_sd, model_sd)

    def test_model_weight_averaging_with_nan_metric(self):
        corrupted_metric_values = np.nan, +np.inf, -np.inf, torch.nan, torch.inf, -torch.inf

        for corrupted_metric_value in corrupted_metric_values:
            with self.subTest(corrupted_metric_value=corrupted_metric_value):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    avg = ModelWeightAveraging(
                        ckpt_dir=tmp_dir,
                        greater_is_better=True,
                        metric_to_watch="acc",
                        load_checkpoint=False,
                        number_of_models_to_average=10,
                    )

                    model = self._create_dummy_model()
                    model_sd = model.state_dict()
                    avg.get_average_model(model, {"acc": 0.99})

                    corrupted_model = self._create_dummy_model()
                    corrupted_model.fc1.weight.data = torch.randn(10, 10) * torch.nan
                    avg_model_sd = avg.get_average_model(corrupted_model, {"acc": corrupted_metric_value})

                    self.assertStateDictAlmostEqual(avg_model_sd, model_sd)

    def assertStateDictAlmostEqual(self, sd1, sd2, eps=1e-5):
        self.assertEqual(set(sd1.keys()), set(sd2.keys()))
        for key in sd1.keys():
            v1 = sd1[key]
            v2 = sd2[key]
            if torch.is_floating_point(v1) and torch.is_floating_point(v2):
                difference = torch.nn.functional.l1_loss(v1, v2)
                self.assertLessEqual(difference, eps, msg=f"{key}: {v1} vs {v2}")
            else:
                self.assertEqual(v1, v2)

    def _create_dummy_model(self) -> nn.Module:
        net = nn.Sequential(collections.OrderedDict([("fc1", nn.Linear(10, 10)), ("bn", nn.BatchNorm1d(10))]))
        net.fc1.weight.data = torch.randn(10, 10)
        return net


if __name__ == "__main__":
    unittest.main()
