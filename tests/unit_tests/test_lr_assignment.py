import unittest

from super_gradients.common.object_names import Models
from super_gradients.training.models import LeNet
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils.optimizer_utils import separate_lr_groups
from super_gradients.training import models


class TestSeparateLRGroups(unittest.TestCase):
    def test_all_parameters_covered(self):
        model = LeNet()  # Create your model
        lr_dict = {"fc3": 0.01, "fc2": 0.001, "fc1": 0.005, "default": 0.1}

        param_groups = separate_lr_groups(model, lr_dict)

        all_params = set()
        for group in param_groups:
            all_params.update(param[0] for param in group["named_params"])

        all_named_params = set(param[0] for param in model.named_parameters())

        self.assertEqual(all_params, all_named_params)

    def test_no_parameter_intersection(self):
        model = LeNet()  # Create your model
        lr_dict = {"head": 0.01, "backbone": 0.001, "compression3": 0.005, "default": 0.1}

        param_groups = separate_lr_groups(model, lr_dict)

        for group1 in param_groups:
            for group2 in param_groups:
                if group1 != group2:
                    intersection = set(param[0] for param in group1["named_params"]).intersection(set(param[0] for param in group2["named_params"]))
                    self.assertEqual(len(intersection), 0)

    def test_ddrnet_param_groups_consistency(self):
        model = models.get(Models.DDRNET_23, pretrained_weights="cityscapes")
        lr_dict = {
            "default": 0.075,
            # backbone layers
            "_backbone": 0.0075,
            "compression3": 0.0075,
            "compression4": 0.0075,
            "down3": 0.0075,
            "down4": 0.0075,
            "layer3_skip": 0.0075,
            "layer4_skip": 0.0075,
            "layer5_skip": 0.0075,
        }

        param_groups = separate_lr_groups(model, lr_dict)
        param_groups_old = model.initialize_param_groups(0.0075, training_params=HpmStruct(multiply_head_lr=10))

        self._check_param_groups_assign_same_lrs(param_groups, param_groups_old)

    def test_ppliteseg_param_groups_consistency(self):
        model = models.get(Models.PP_LITE_T_SEG50, pretrained_weights="cityscapes")
        lr_dict = {"encoder.backbone": 0.01, "default": 0.1}

        param_groups = separate_lr_groups(model, lr_dict)
        param_groups_old = model.initialize_param_groups(0.01, training_params=HpmStruct(multiply_head_lr=10))

        self._check_param_groups_assign_same_lrs(param_groups, param_groups_old)

    def test_stdc_param_groups_consistency(self):
        model = models.get(Models.STDC1_SEG50, pretrained_weights="cityscapes")
        lr_dict = {"cp": 0.005, "default": 0.05}

        param_groups = separate_lr_groups(model, lr_dict)
        param_groups_old = model.initialize_param_groups(0.005, training_params=HpmStruct(multiply_head_lr=10, loss=None))

        self._check_param_groups_assign_same_lrs(param_groups, param_groups_old)

    def test_regseg_param_groups_consistency(self):
        model = models.get(Models.REGSEG48, pretrained_weights="cityscapes")
        lr_dict = {"head.": 0.05, "default": 0.005}

        param_groups = separate_lr_groups(model, lr_dict)
        param_groups_old = model.initialize_param_groups(0.005, training_params=HpmStruct(multiply_head_lr=10, loss=None))

        self._check_param_groups_assign_same_lrs(param_groups, param_groups_old)

    def test_segformer_param_groups_consistency(self):
        model = models.get(Models.SEGFORMER_B0, pretrained_weights="cityscapes")
        lr_dict = {"default": 0.05, "_backbone": 0.005}

        param_groups = separate_lr_groups(model, lr_dict)
        param_groups_old = model.initialize_param_groups(0.005, training_params=HpmStruct(multiply_head_lr=10, loss=None))

        self._check_param_groups_assign_same_lrs(param_groups, param_groups_old)

    def _check_param_groups_assign_same_lrs(self, param_groups, param_groups_old):
        names_lr_pairs = set([(sub_group[0], group["lr"]) for group in param_groups for sub_group in group["named_params"]])
        names_lr_pairs_old = set([(sub_group[0], group["lr"]) for group in param_groups_old for sub_group in group["named_params"]])
        self.assertEqual(set(names_lr_pairs_old), set(names_lr_pairs))

    if __name__ == "__main__":
        unittest.main()


if __name__ == "__main__":
    unittest.main()
