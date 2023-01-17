import unittest
import torch.nn as nn
import torch.nn.functional as F
import torch

from super_gradients.training.utils.optimizer_utils import separate_zero_wd_params_groups_for_optimizer
from super_gradients.training.utils import HpmStruct
from super_gradients.training.models.sg_module import SgModule


class ToyLinearKernel(nn.Module):
    """
    Custom Toy linear module to test custom modules with bias parameter, that are not instances of primitive torch
    modules.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor):
        return F.linear(input, self.weight, self.bias)


class ToySgModule(SgModule):
    """
    Toy Module to test zero of weight decay, support multiple group of parameters.
    """

    CONV_CLASSES = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    CONV_TRANSPOSE_CLASSES = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
    BN_CLASSES = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    def __init__(self, input_dimension=2, multiple_param_groups=False, module_groups=False, linear_cls=nn.Linear):
        """
        :param input_dimension: input dimension, 1 for 1D, 2 for 2D ...
        :param multiple_param_groups: if True create multiple param groups with different optimizer args.
        """
        super().__init__()
        num_classes = 10

        self.multiple_param_groups = multiple_param_groups
        self.module_groups = module_groups

        self.conv_cls = self.CONV_CLASSES[input_dimension]
        self.bn_cls = self.BN_CLASSES[input_dimension]
        self.conv_transpose_cls = self.CONV_TRANSPOSE_CLASSES[input_dimension]
        self.linear_cls = linear_cls

        self.num_conv = 0
        self.num_bn = 0
        self.num_biases = 0
        self.num_linear = 0

        self.base = nn.Sequential(
            self.conv1(3, 128, 2),
            self.conv1(128, 128, 2, bias=True),
            self.conv_transpose(128, 128),
            self.conv_transpose(128, 128, bias=True),
        )

        self.base_params = (self.num_no_decay_params(), self.num_decay_params())

        self.more_convs = nn.Sequential(
            self.conv1(128, 128, 1),
            self.conv1(128, 128, 2, bias=True),
            self.conv_transpose(128, 128),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(self.linear(128, 2 * num_classes, bias=False), self.linear(2 * num_classes, num_classes, bias=True))

        self.head = nn.Sequential(self.more_convs, self.avg_pool, self.classifier)

        self.head_params = (self.num_no_decay_params() - self.base_params[0], self.num_decay_params() - self.base_params[1])

    def conv1(self, ch_in: int, ch_out: int, stride: int, bias=False):
        self.num_conv += 1
        if bias:
            conv = self.conv_cls(ch_in, ch_out, 1, stride=stride, bias=bias)
            self.num_biases += 1
        else:
            conv = nn.Sequential(self.conv_cls(ch_in, ch_out, 1, stride=stride, bias=bias), self.bn_cls(ch_out), nn.ReLU())
            self.num_bn += 1
        return conv

    def conv_transpose(self, ch_in: int, ch_out: int, bias=False):
        self.num_conv += 1
        if bias:
            conv = self.conv_transpose_cls(ch_in, ch_out, 2, stride=2, bias=bias)
            self.num_biases += 1
        else:
            conv = nn.Sequential(self.conv_transpose_cls(ch_in, ch_out, 2, stride=2, bias=bias), self.bn_cls(ch_out), nn.ReLU())
            self.num_bn += 1
        return conv

    def linear(self, ch_in: int, ch_out: int, bias=False):
        self.num_linear += 1
        if bias:
            self.num_biases += 1
        return self.linear_cls(ch_in, ch_out, bias)

    def num_decay_params(self):
        return self.num_conv + self.num_linear

    def num_no_decay_params(self):
        return self.num_biases + 2 * self.num_bn

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        # Example to different learning rates, similar to ShelfNet, in order to create multiple groups.
        if self.multiple_param_groups:
            params_list = [{"named_params": self.base.named_parameters(), "lr": lr}, {"named_params": self.head.named_parameters(), "lr": lr * 10}]

            return params_list

        return super().initialize_param_groups(lr, training_params)


class ZeroWdForBnBiasTest(unittest.TestCase):
    """
    Testing if the optimizer parameters are divided into two groups with one being with weight_decay = 0
    """

    def setUp(self):
        # Define Parameters
        self.weight_decay = 0.01
        self.lr = 0.1
        # input dimension beside batch and channels, i.e 2 for vision, 1 for audio, 3 for point cloud.
        self.input_dimensions = (1, 2, 3)
        self.train_params_zero_wd = {"initial_lr": self.lr, "optimizer": "SGD", "optimizer_params": {"weight_decay": self.weight_decay, "momentum": 0.9}}

    def _assert_optimizer_param_groups(
        self, param_groups: list, excpected_num_groups: int, excpected_num_params_per_group: list, excpected_weight_decay_per_group: list
    ):
        """
        Helper method to assert, num of param_groups, num of parameters in each param group and weight decay value
        in each param group.
        """
        self.assertEqual(len(param_groups), excpected_num_groups, msg=f"Optimizer should have {excpected_num_groups} groups")
        for (param_group, excpected_num_params, excpected_weight_decay) in zip(param_groups, excpected_num_params_per_group, excpected_weight_decay_per_group):
            self.assertEqual(
                len(param_group["params"]),
                excpected_num_params,
                msg="Wrong number of params for optimizer param group, excpected: {}, found: {}".format(excpected_num_params, len(param_group["params"])),
            )
            self.assertEqual(
                param_group["weight_decay"],
                excpected_weight_decay,
                msg="Wrong weight decay value found for optimizer param group, excpected: {}, found: {}".format(
                    excpected_weight_decay, param_group["weight_decay"]
                ),
            )

    def test_zero_wd_one_group(self):
        """
        test that one group of parameters are separated to weight_decay_params and without.
        """
        for input_dim in self.input_dimensions:
            net = ToySgModule(input_dimension=input_dim)
            train_params = HpmStruct(**self.train_params_zero_wd)

            optimizer_params_groups = separate_zero_wd_params_groups_for_optimizer(net, net.initialize_param_groups(self.lr, train_params), self.weight_decay)

            self._assert_optimizer_param_groups(
                optimizer_params_groups,
                excpected_num_groups=2,
                excpected_num_params_per_group=[net.num_no_decay_params(), net.num_decay_params()],
                excpected_weight_decay_per_group=[0, self.weight_decay],
            )

    def test_zero_wd_multiple_group(self):
        """
        test that 2 groups of parameters are separated to 2 groups of weight_decay_params and 2 groups without.
        """
        for input_dim in self.input_dimensions:
            net = ToySgModule(input_dimension=input_dim, multiple_param_groups=True)
            train_params = HpmStruct(**self.train_params_zero_wd)

            optimizer_params_groups = separate_zero_wd_params_groups_for_optimizer(net, net.initialize_param_groups(self.lr, train_params), self.weight_decay)

            self._assert_optimizer_param_groups(
                optimizer_params_groups,
                excpected_num_groups=4,
                excpected_num_params_per_group=[net.base_params[0], net.base_params[1], net.head_params[0], net.head_params[1]],
                excpected_weight_decay_per_group=[0, self.weight_decay, 0, self.weight_decay],
            )

    def test_zero_wd_sync_bn(self):
        """
        test affiliation of nn.SyncBatchNorm parameters to zero weight decay.
        """
        for input_dim in self.input_dimensions:
            net = ToySgModule(input_dimension=input_dim)
            # Convert to SyncBatchNorm
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

            train_params = HpmStruct(**self.train_params_zero_wd)

            optimizer_params_groups = separate_zero_wd_params_groups_for_optimizer(net, net.initialize_param_groups(self.lr, train_params), self.weight_decay)

            self._assert_optimizer_param_groups(
                optimizer_params_groups,
                excpected_num_groups=2,
                excpected_num_params_per_group=[net.num_no_decay_params(), net.num_decay_params()],
                excpected_weight_decay_per_group=[0, self.weight_decay],
            )

    def test_zero_wd_custom_module_with_bias(self):
        """
        test affiliation of nn.SyncBatchNorm parameters to zero weight decay.
        """
        input_dim = 2
        net = ToySgModule(input_dimension=input_dim, linear_cls=ToyLinearKernel)

        train_params = HpmStruct(**self.train_params_zero_wd)

        optimizer_params_groups = separate_zero_wd_params_groups_for_optimizer(net, net.initialize_param_groups(self.lr, train_params), self.weight_decay)

        self._assert_optimizer_param_groups(
            optimizer_params_groups,
            excpected_num_groups=2,
            excpected_num_params_per_group=[net.num_no_decay_params(), net.num_decay_params()],
            excpected_weight_decay_per_group=[0, self.weight_decay],
        )


if __name__ == "__main__":
    unittest.main()
