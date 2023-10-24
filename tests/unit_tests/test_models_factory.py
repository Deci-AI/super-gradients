import unittest

import torch
from super_gradients.training import models

# This is a subset of all the models, since some cannot be instantiated with models.get() without explicit arch_params
MODELS = [
    "vit_base",
    "vit_large",
    "vit_huge",
    "beit_base_patch16_224",
    "beit_large_patch16_224",
    "custom_densenet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_b8",
    "efficientnet_l2",
    "mobilenet_v2",
    "mobile_net_v2_135",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "resnet18",
    "resnet18_cifar",
    "resnet34",
    "resnet50",
    "resnet50_3343",
    "resnet101",
    "resnet152",
    "resnext50",
    "resnext101",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "csp_darknet53",
    "ppyoloe_s",
    "ppyoloe_m",
    "ppyoloe_l",
    "ppyoloe_x",
    "darknet53",
    "ssd_mobilenet_v1",
    "ssd_lite_mobilenet_v2",
    "regnetY200",
    "regnetY400",
    "regnetY600",
    "regnetY800",
    "yolox_n",
    "yolox_t",
    "yolox_s",
    "yolox_m",
    "yolox_l",
    "yolox_x",
    "yolo_nas_s",
    "yolo_nas_m",
    "yolo_nas_l",
    "shelfnet18_lw",
    "shelfnet34_lw",
    # "shelfnet50_3343", # FIXME: seems to not work correctly
    # "shelfnet50", # FIXME: seems to not work correctly
    # "shelfnet101", # FIXME: seems to not work correctly
    "stdc1_classification",
    "stdc2_classification",
    "stdc1_seg75",
    "stdc1_seg50",
    "stdc1_seg",
    "stdc2_seg75",
    "stdc2_seg50",
    "stdc2_seg",
    "ddrnet_39",
    "ddrnet_23",
    "ddrnet_23_slim",
    "pp_lite_b_seg75",
    "pp_lite_b_seg50",
    "pp_lite_b_seg",
    "pp_lite_t_seg75",
    "pp_lite_t_seg50",
    "pp_lite_t_seg",
    "regseg48",
    "segformer_b0",
    "segformer_b1",
    "segformer_b2",
    "segformer_b3",
    "segformer_b4",
    "segformer_b5",
    "dekr_w32_no_dc",
    "yolo_nas_pose_n",
    "yolo_nas_pose_s",
    "yolo_nas_pose_m",
    "yolo_nas_pose_l",
]


def can_model_forward(model, input_channels: int) -> bool:
    """Checks if the given model can perform a forward pass on inputs of certain sizes."""
    input_sizes = [(224, 224), (512, 512)]  # We check different sizes because some model only support one or the other

    for h, w in input_sizes:
        try:
            model(torch.rand(2, input_channels, h, w))
            return True
        except Exception:
            continue

    return False


class DynamicModelTests(unittest.TestCase):
    def test_models(self):
        # TODO: replace `MODELS` with `ARCHITECTURES.keys()` once all models can be instantiated with
        # TODO  models.get() without explicit arch_params without any explicit arch_params

        for model_name in MODELS:
            with self.subTest(model_name=model_name):

                model = models.get(model_name, num_classes=20, num_input_channels=4)
                self.assertEqual(model.get_input_channels(), 4)
                self.assertTrue(can_model_forward(model=model, input_channels=4))

                model.replace_input_channels(51)
                self.assertEqual(model.get_input_channels(), 51)
                self.assertTrue(can_model_forward(model=model, input_channels=51))


if __name__ == "__main__":
    unittest.main()
