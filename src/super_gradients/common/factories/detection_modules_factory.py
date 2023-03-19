from typing import Union, Any

from omegaconf import DictConfig

from super_gradients.training.utils import HpmStruct
from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.registry.registry import ALL_DETECTION_MODULES


class DetectionModulesFactory(BaseFactory):
    def __init__(self):
        super().__init__(ALL_DETECTION_MODULES)

    @staticmethod
    def insert_module_param(conf: Union[str, dict, HpmStruct, DictConfig], name: str, value: Any):
        """
        Assign a new parameter for the module
        :param conf:    a module config, either {type_name(str): {parameters...}} or just type_name(str)
        :param name:    parameter name
        :param value:   parameter value
        :return:        an update config {type_name(str): {name: value, parameters...}}
        """
        if isinstance(conf, str):
            return {conf: {name: value}}

        cls_type = list(conf.keys())[0]
        conf[cls_type][name] = value
        return conf
