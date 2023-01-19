import unittest
import os
import torch

from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path

CKPT_ROOT_DIR = "~/recipe_test/checkpoints"


class ShortenedRecipesAccuracyTests(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.experiment_names = ["shortened_cifar10_resnet_accuracy_test", "shortened_coco2017_yolox_n_map_test", "shortened_cityscapes_regseg48_iou_test"]

    def test_shortened_cifar10_resnet_accuracy(self):
        self.assertTrue(self._reached_goal_metric(experiment_name="shortened_cifar10_resnet_accuracy_test", metric_value=0.9167, delta=0.05))

    def test_convert_shortened_cifar10_resnet(self):
        ckpt_dir = get_checkpoints_dir_path(experiment_name="shortened_cifar10_resnet_accuracy_test", ckpt_root_dir=CKPT_ROOT_DIR)
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "ckpt_best.onnx")))

    def test_shortened_coco2017_yolox_n_map(self):
        self.assertTrue(self._reached_goal_metric(experiment_name="shortened_coco2017_yolox_n_map_test", metric_value=0.044, delta=0.02))

    def test_shortened_cityscapes_regseg48_iou(self):
        self.assertTrue(self._reached_goal_metric(experiment_name="shortened_cityscapes_regseg48_iou_test", metric_value=0.263, delta=0.05))

    @classmethod
    def _reached_goal_metric(cls, experiment_name: str, metric_value: float, delta: float):
        checkpoints_dir_path = get_checkpoints_dir_path(experiment_name=experiment_name, ckpt_root_dir=CKPT_ROOT_DIR)
        sd = torch.load(os.path.join(checkpoints_dir_path, "ckpt_best.pth"))
        metric_val_reached = sd["acc"].cpu().item()
        diff = abs(metric_val_reached - metric_value)
        print(
            "Goal metric value: " + str(metric_value) + ", metric value reached: " + str(metric_val_reached) + ",diff: " + str(diff) + ", delta: " + str(delta)
        )
        return diff <= delta


if __name__ == "__main__":
    unittest.main()
