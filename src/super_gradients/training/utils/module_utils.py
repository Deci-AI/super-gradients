from collections import OrderedDict
import copy
from typing import List, Union, Optional
import torch
from torch import nn
from omegaconf.listconfig import ListConfig

from super_gradients.common import UpsampleMode


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
        self._modules["0"] = module
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
        self._modules["0"](x)
        outputs = self._outputs_lists[x.device]
        self._outputs_lists = {}
        return outputs

    def _get_recursive(self, module: nn.Module, path) -> nn.Module:
        """recursively look for a module using a path"""
        if not isinstance(path, (list, ListConfig)):
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
            key = path[0] if isinstance(path, (list, ListConfig)) else path
            index = list(module._modules).index(str(key))
            if index > last_index:
                last_index = index
                last_key = key

        module._modules = self._slice_odict(module._modules, 0, last_index + 1)

        next_level_paths = []
        for path in output_paths:
            if isinstance(path, (list, ListConfig)) and path[0] == last_key and len(path) > 1:
                next_level_paths.append(path[1:])

        if len(next_level_paths) > 0:
            self._prune(module._modules[str(last_key)], next_level_paths)

    def _slice_odict(self, odict: OrderedDict, start: int, end: int):
        """Slice an OrderedDict in the same logic list,tuple... are sliced"""
        return OrderedDict([(k, v) for (k, v) in odict.items() if k in list(odict.keys())[start:end]])


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
    assert isinstance(new_activation, nn.Module), "new_activation should be nn.Module"
    assert all(
        [isinstance(t, type) and issubclass(t, nn.Module) for t in activations_to_replace]
    ), "activations_to_replace should be types that are subclasses of nn.Module"

    # do the replacement
    _replace_activations_recursive(module, new_activation, activations_to_replace)


def fuse_repvgg_blocks_residual_branches(model: nn.Module):
    """
    Call fuse_block_residual_branches for all repvgg blocks in the model
    :param model: torch.nn.Module with repvgg blocks. Doesn't have to be entirely consists of repvgg.
    :type model: torch.nn.Module
    """
    assert not model.training, "To fuse RepVGG block residual branches, model must be on eval mode"
    for module in model.modules():
        if hasattr(module, "fuse_block_residual_branches"):
            module.fuse_block_residual_branches()
    model.build_residual_branches = False


class NormalizationAdapter(torch.nn.Module):
    """
    Denormalizes input by mean_original, std_original, then normalizes by mean_required, std_required.

    Used in KD training where teacher expects data normalized by mean_required, std_required.

    mean_original, std_original, mean_required, std_required are all list-like objects of length that's equal to the
     number of input channels.

    """

    def __init__(self, mean_original, std_original, mean_required, std_required):
        super(NormalizationAdapter, self).__init__()
        mean_original = torch.tensor(mean_original).unsqueeze(-1).unsqueeze(-1)
        std_original = torch.tensor(std_original).unsqueeze(-1).unsqueeze(-1)
        mean_required = torch.tensor(mean_required).unsqueeze(-1).unsqueeze(-1)
        std_required = torch.tensor(std_required).unsqueeze(-1).unsqueeze(-1)

        self.additive = torch.nn.Parameter((mean_original - mean_required) / std_original)
        self.multiplier = torch.nn.Parameter(std_original / std_required)

    def forward(self, x):
        x = (x + self.additive) * self.multiplier
        return x


def make_upsample_module(scale_factor: int, upsample_mode: Union[str, UpsampleMode], align_corners: Optional[bool] = None):
    """
    Factory method for creating upsampling modules.
    :param scale_factor: upsample scale factor
    :param upsample_mode: see UpsampleMode for supported options.
    :return: nn.Module
    """
    upsample_mode = upsample_mode.value if isinstance(upsample_mode, UpsampleMode) else upsample_mode
    if upsample_mode == UpsampleMode.NEAREST.value:
        # Prevent ValueError when passing align_corners with nearest mode.
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode)
    elif upsample_mode in [UpsampleMode.BILINEAR.value, UpsampleMode.BICUBIC.value]:
        module = nn.Upsample(scale_factor=scale_factor, mode=upsample_mode, align_corners=align_corners)
    else:
        raise NotImplementedError(f"Upsample mode: `{upsample_mode}` is not supported.")
    return module
