from typing import Any, Dict, List, Union

from omegaconf import DictConfig

from super_gradients.training.utils.utils import HpmStruct


class DetectionModulesFactory:
    """
    The factory for creating a sub-module of a detection model according to the following scheme:
    a module excepts arch_params and in_channels,
    where in_channels defines channels of tensor(s) that will be accepted by a module in forward
    """

    def __init__(self, type_dict: Dict[str, type]):
        """
        :param type_dict: a dictionary mapping a type name to an actual type
        """
        self.type_dict = type_dict

    def get(self, arch_params: Union[DictConfig, HpmStruct, Dict[str, Any]], in_channels: Union[int, List[int]]):
        """
        Get an instantiated module
        :param arch_params: a configuration {'type': 'type_name', other_arch_params... }}
        :param in_channels: will be passed into the module during construction
        """
        module_type = arch_params['type']
        if module_type not in self.type_dict:
            raise RuntimeError(f'Unknown object type: {module_type} in configuration. '
                               f'Valid types are: {self.type_dict.keys()}')

        arch_params['factory'] = self
        return self.type_dict[module_type](arch_params, in_channels)
