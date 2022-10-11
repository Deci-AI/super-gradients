import importlib
from typing import Union

from super_gradients.common.factories.base_factory import AbstractFactory
import super_gradients.training.models.segmentation_models.context_modules as context_modules


class ContextsTypeFactory(AbstractFactory):
    def __init__(self):
        self.type_dict = context_modules.CONTEXT_TYPE_DICT

    def get(self, conf: Union[str]):
        """
         Get a type.
            :param conf: a configuration
            if string - assumed to be a type name (not the real name, but a name defined in the Factory)
            a dictionary is not supported, since the actual instantiation takes place elsewhere,
            due to dynamic in_channels argument.
        """
        if isinstance(conf, str):
            if conf in self.type_dict:
                return self.type_dict[conf]
            else:
                try:
                    lib = '.'.join(conf.split('.')[:-1])
                    module = conf.split('.')[-1]
                    lib = importlib.import_module(lib)  # Import the required packages
                    class_type = lib.__dict__[module]
                    return class_type
                except RuntimeError:
                    raise RuntimeError(f"Unknown object type: {conf} in configuration. valid types are: {self.type_dict.keys()} or a class "
                                       "type available in the env (or the form 'package_name.sub_package.MyClass'")
        else:
            return conf
