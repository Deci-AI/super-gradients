import unittest
import shutil

from coverage.annotate import os
from super_gradients.common.environment import environment_config
import torch


class ShortenedRecipesAccuracyTests(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.experiment_names = ["shortened_cifar10_resnet_accuracy_test", "shortened_coco2017_yolox_n_map_test", "shortened_cityscapes_regseg48_iou_test"]

    def test_shortened_cifar10_resnet_accuracy(self):
        self.assertTrue(self._reached_goal_metric(experiment_name="shortened_cifar10_resnet_accuracy_test", metric_value=0.9167, delta=0.05))

    def test_shortened_coco2017_yolox_n_map(self):
        self.assertTrue(self._reached_goal_metric(experiment_name="shortened_coco2017_yolox_n_map_test", metric_value=0.044, delta=0.02))

    def test_shortened_cityscapes_regseg48_iou(self):
        self.assertTrue(self._reached_goal_metric(experiment_name="shortened_cityscapes_regseg48_iou_test", metric_value=0.263, delta=0.05))

    @classmethod
    def _reached_goal_metric(cls, experiment_name: str, metric_value: float, delta: float):
        ckpt_dir = os.path.join(environment_config.PKG_CHECKPOINTS_DIR, experiment_name)
        sd = torch.load(os.path.join(ckpt_dir, "ckpt_best.pth"))
        metric_val_reached = sd["acc"].cpu().item()
        diff = abs(metric_val_reached - metric_value)
        print(
            "Goal metric value: " + str(metric_value) + ", metric value reached: " + str(metric_val_reached) + ",diff: " + str(diff) + ", delta: " + str(delta)
        )
        return diff <= delta

    @classmethod
    def tearDownClass(cls) -> None:
        # ERASE ALL THE FOLDERS THAT WERE CREATED DURING THIS TEST
        for folder in cls.experiment_names:
            ckpt_dir = os.path.join(environment_config.PKG_CHECKPOINTS_DIR, folder)
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)


if __name__ == "__main__":
    unittest.main()
