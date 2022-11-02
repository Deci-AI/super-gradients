from typing import Dict, Union, Type
from enum import Enum
import importlib

from super_gradients.common.factories.base_factory import AbstractFactory


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
        if isinstance(conf, str):
            if conf in self.type_dict:
                return self.type_dict[conf]
            else:
                try:
                    lib = ".".join(conf.split(".")[:-1])
                    module = conf.split(".")[-1]
                    lib = importlib.import_module(lib)  # Import the required packages
                    class_type = lib.__dict__[module]
                    return class_type
                except RuntimeError:
                    raise RuntimeError(
                        f"Unknown object type: {conf} in configuration. valid types are: {self.type_dict.keys()} or a class "
                        "type available in the env (or the form 'package_name.sub_package.MyClass'"
                    )
        else:
            return conf
