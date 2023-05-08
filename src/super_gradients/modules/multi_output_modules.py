from collections import OrderedDict
from torch import nn
from omegaconf.listconfig import ListConfig


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
