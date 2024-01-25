import unittest
from copy import deepcopy

from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.models import LeNet
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils.optimizer_utils import separate_lr_groups
from super_gradients.training import models
from super_gradients.training.utils.utils import check_models_have_same_weights


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
        lr_dict = {"fc3": 0.01, "fc2": 0.001, "fc1": 0.005, "default": 0.1}

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

    def test_requires_grad_false(self):
        # Test when some layers have requires_grad==False
        model = LeNet()
        lr_dict = {"fc2": 0.001, "fc1": 0.005, "default": 0.1}
        for param in model.fc3.parameters():
            param.requires_grad = False

        param_groups = separate_lr_groups(model, lr_dict)
        total_optimizable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Extract tensors from the "named_params" entry in each dictionary
        tensors_in_param_groups = [entry[1] for group in param_groups for entry in group["named_params"]]
        total_params_in_param_groups = sum(t.numel() for t in tensors_in_param_groups)

        self.assertEqual(total_params_in_param_groups, total_optimizable_params)

    def test_initial_lr_zero(self):
        # Test case when initial_lr = {"default": 1, "some_layer": 0}
        model = LeNet()
        lr_dict = {
            "default": 1,
            "fc1": 0,
        }

        param_groups = separate_lr_groups(model, lr_dict)
        total_non_optimizable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total_model_params = sum(p.numel() for p in model.parameters())

        # Extract tensors from the "named_params" entry in each dictionary
        tensors_in_param_groups = [entry[1] for group in param_groups for entry in group["named_params"]]
        total_params_in_param_groups = sum(t.numel() for t in tensors_in_param_groups)

        self.assertEqual(total_params_in_param_groups, total_model_params - total_non_optimizable_params)

    def test_train_with_lr_assignment(self):
        # Define Model
        net = LeNet()
        net_before_train = deepcopy(net)

        trainer = Trainer("test_train_with_lr_assignment")

        train_params = {
            "max_epochs": 3,
            "lr_decay_factor": 0.1,
            "initial_lr": {
                "default": 0,
                "fc3": 0.1,
            },
            "loss": "CrossEntropyLoss",
            "optimizer": "SGD",
            "criterion_params": {},
            "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
            "train_metrics_list": [Accuracy()],
            "valid_metrics_list": [Accuracy()],
            "metric_to_watch": "Accuracy",
            "greater_metric_to_watch_is_better": True,
            "ema": False,
            "phase_callbacks": [],
        }

        trainer.train(
            model=net,
            training_params=train_params,
            train_loader=classification_test_dataloader(batch_size=4),
            valid_loader=classification_test_dataloader(batch_size=4),
        )

        self.assertTrue(check_models_have_same_weights(net_before_train.conv1, net.conv1))
        self.assertTrue(check_models_have_same_weights(net_before_train.conv2, net.conv2))
        self.assertTrue(check_models_have_same_weights(net_before_train.fc1, net.fc1))
        self.assertTrue(check_models_have_same_weights(net_before_train.fc2, net.fc2))
        self.assertFalse(check_models_have_same_weights(net_before_train.fc3, net.fc3))

    def _check_param_groups_assign_same_lrs(self, param_groups, param_groups_old):
        names_lr_pairs = set([(sub_group[0], group["lr"]) for group in param_groups for sub_group in group["named_params"]])
        names_lr_pairs_old = set([(sub_group[0], group["lr"]) for group in param_groups_old for sub_group in group["named_params"]])
        self.assertEqual(set(names_lr_pairs_old), set(names_lr_pairs))

    if __name__ == "__main__":
        unittest.main()


if __name__ == "__main__":
    unittest.main()
