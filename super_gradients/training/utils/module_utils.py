from collections import OrderedDict
import copy
from typing import List, Union, Tuple

from torch import nn


class MultiOutputModule(nn.Module):
    """
    This module wraps around a container nn.Module (such as Module, Sequential and ModuleList) and allows to extract
    multiple output from its inner modules on each forward call() (as a list of output tensors)
    note: the default output of the wrapped module will not be added to the output list by default. To get
    the default output in the outputs list, explicitly include its path in the @output_paths parameter

    i.e.
    for module:
        Sequential(
          (0): Sequential(
            (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )                                         ===================================>>
          (1): InvertedResidual(
            (conv): Sequential(
              (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)              ===================================>>
              (3): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
    and paths:
        [0, [1, 'conv', 2]]
    the output are marked with arrows
    """

    def __init__(self, module: nn.Module, output_paths: list, prune: bool = True):
        """
        :param module: The wrapped container module
        :param output_paths: a list of lists or keys containing the canonical paths to the outputs
        i.e. [3, [4, 'conv', 5], 7] will extract outputs of layers 3, 7 and 4->conv->5
        """
        super().__init__()
        self.output_paths = output_paths
        self._modules['0'] = module
        self._outputs_lists = {}

        for path in output_paths:
            child = self._get_recursive(module, path)
            child.register_forward_hook(hook=self.save_output_hook)

        # PRUNE THE MODULE TO SUPPORT ALL PROVIDED OUTPUT_PATHS BUT REMOVE ALL REDUNDANT LAYERS
        if prune:
            self._prune(module, output_paths)

    def save_output_hook(self, _, input, output):
        self._outputs_lists[input[0].device].append(output)

    def forward(self, x) -> list:
        self._outputs_lists[x.device] = []
        self._modules['0'](x)
        return self._outputs_lists[x.device]

    def _get_recursive(self, module: nn.Module, path) -> nn.Module:
        """recursively look for a module using a path"""
        if not isinstance(path, list):
            return module._modules[str(path)]
        elif len(path) == 1:
            return module._modules[str(path[0])]
        else:
            return self._get_recursive(module._modules[str(path[0])], path[1:])

    def _prune(self, module: nn.Module, output_paths: list):
        """
        Recursively prune the module to support all provided output_paths but remove all redundant layers
        """
        last_index = -1
        last_key = None

        # look for the last key from all paths
        for path in output_paths:
            key = path[0] if isinstance(path, list) else path
            index = list(module._modules).index(str(key))
            if index > last_index:
                last_index = index
                last_key = key

        module._modules = self._slice_odict(module._modules, 0, last_index + 1)

        next_level_paths = []
        for path in output_paths:
            if isinstance(path, list) and path[0] == last_key and len(path) > 1:
                next_level_paths.append(path[1:])

        if len(next_level_paths) > 0:
            self._prune(module._modules[str(last_key)], next_level_paths)

    def _slice_odict(self, odict: OrderedDict, start: int, end: int):
        """Slice an OrderedDict in the same logic list,tuple... are sliced"""
        return OrderedDict([
            (k, v) for (k, v) in odict.items()
            if k in list(odict.keys())[start:end]
        ])


def _replace_activations_recursive(module: nn.Module, new_activation: nn.Module, activations_to_replace: List[type]):
    """
    A helper called in replace_activations(...)
    """
    for n, m in module.named_children():
        if type(m) in activations_to_replace:
            setattr(module, n, copy.deepcopy(new_activation))
        else:
            _replace_activations_recursive(m, new_activation, activations_to_replace)


def replace_activations(module: nn.Module, new_activation: nn.Module, activations_to_replace: List[type]):
    """
    Recursively go through module and replaces each activation in activations_to_replace with a copy of new_activation
    :param module:                  a module that will be changed inplace
    :param new_activation:          a sample of a new activation (will be copied)
    :param activations_to_replace:  types of activations to replace, each must be a subclass of nn.Module
    """
    # check arguments once before the recursion
    assert isinstance(new_activation, nn.Module), 'new_activation should be nn.Module'
    assert all([isinstance(t, type) and issubclass(t, nn.Module) for t in activations_to_replace]), \
        'activations_to_replace should be types that are subclasses of nn.Module'

    # do the replacement
    _replace_activations_recursive(module, new_activation, activations_to_replace)


def fuse_repvgg_blocks_residual_branches(model: nn.Module):
    '''
    Call fuse_block_residual_branches for all repvgg blocks in the model
    :param model: torch.nn.Module with repvgg blocks. Doesn't have to be entirely consists of repvgg.
    :type model: torch.nn.Module
    '''
    assert not model.training, "To fuse RepVGG block residual branches, model must be on eval mode"
    for module in model.modules():
        if hasattr(module, 'fuse_block_residual_branches'):
            module.fuse_block_residual_branches()
    model.build_residual_branches = False


class ConvBNReLU(nn.Module):
    """
    Class for Convolution2d-Batchnorm2d-Relu layer. Default behaviour is Conv-BN-Relu. To exclude Batchnorm module use
        `use_normalization=False`, to exclude Relu activation use `use_activation=False`.
    For convolution arguments documentation see `nn.Conv2d`.
    For batchnorm arguments documentation see `nn.BatchNorm2d`.
    For relu arguments documentation see `nn.Relu`.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 use_normalization: bool = True,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None,
                 use_activation: bool = True,
                 inplace: bool = False):

        super(ConvBNReLU, self).__init__()
        self.seq = nn.Sequential()
        self.seq.add_module("conv", nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=bias,
                                              padding_mode=padding_mode))

        if use_normalization:
            self.seq.add_module("bn", nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=affine,
                                                     track_running_stats=track_running_stats, device=device,
                                                     dtype=dtype))
        if use_activation:
            self.seq.add_module("relu", nn.ReLU(inplace=inplace))

    def forward(self, x):
        return self.seq(x)
