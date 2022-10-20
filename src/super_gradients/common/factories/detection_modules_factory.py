from typing import Union, Mapping, Dict


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

    def get(self, conf: Union[str, dict, Dict], in_channels: Union[int, List[int]]):
        """
        Get an instantiated module
        :param conf: a configuration, either a type_name(str) or {type_name(str): {arch_params...}}
        :param in_channels: will be passed into the module during construction
        """
        assert isinstance(conf, str) or (isinstance(conf, Mapping) and len(conf) == 1), \
            'Received a wrong config for a module, expected either a string type name (without parameters) ' \
            f'or a mapping of a type name to its arch_params. The config is {conf}'

        module_type = conf if isinstance(conf, str) else list(conf.keys())[0]
        assert  module_type in self.type_dict, f'Unknown object type: {conf} in configuration. ' \
                                         f'Valid types are: {self.type_dict.keys()}'

        if isinstance(conf, str):
            return self.type_dict[module_type](in_channels)
        else:
            arch_params = list(conf.values())[0]
            arch_params['factory'] = self
            return self.type_dict[module_type](arch_params, in_channels)
