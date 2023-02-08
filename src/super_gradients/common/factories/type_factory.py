from typing import Dict, Union, Type
from enum import Enum
import importlib

from super_gradients.common.exceptions.factory_exceptions import UnknownTypeException
from super_gradients.common.factories.base_factory import AbstractFactory
from super_gradients.training.utils import get_param


class TypeFactory(AbstractFactory):
    """
    Factory to return class type from configuration string.
    """

    def __init__(self, type_dict: Dict[str, type]):
        """
        :param type_dict: a dictionary mapping a name to a type
        """
        self.type_dict = type_dict

    @classmethod
    def from_enum_cls(cls, enum_cls: Type[Enum]):
        return cls({entity.name: entity.value for entity in enum_cls})

    def get(self, conf: Union[str, type]):
        """
        Get a type.
           :param conf: a configuration
           if string - assumed to be a type name (not the real name, but a name defined in the Factory)
           a dictionary is not supported, since the actual instantiation takes place elsewhere

           If provided value is already a class type, the value will be returned as is.
        """
        if isinstance(conf, str) or isinstance(conf, bool):
            if conf in self.type_dict:
                return self.type_dict[conf]
            elif isinstance(conf, str) and get_param(self.type_dict, conf) is not None:
                return get_param(self.type_dict, conf)
            elif "." in conf:
                *lib_path, module = conf.split(".")
                lib = ".".join(lib_path)
                try:
                    lib = importlib.import_module(lib)  # Import the required packages
                    class_type = lib.__dict__[module]
                    return class_type
                except Exception as e:
                    err = f"An error occurred while instantiating '{conf}' with exception: \n\t => {e}.\n"
                    raise ValueError(err)
            else:
                raise UnknownTypeException(
                    unknown_type=conf,
                    choices=list(self.type_dict.keys()),
                    message=f"Unknown object type: {conf} in configuration. valid types are: {self.type_dict.keys()} or a class "
                    "type available in the env or in the form 'package_name.sub_package.MyClass'\n",
                )
        else:
            return conf
