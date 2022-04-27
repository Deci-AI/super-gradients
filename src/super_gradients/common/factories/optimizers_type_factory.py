import importlib
from typing import Union

from torch import optim

from super_gradients.common.factories.base_factory import AbstractFactory
from super_gradients.training.utils.optimizers.rmsprop_tf import RMSpropTF
from super_gradients.training.utils.optimizers.lamb import Lamb


class OptimizersTypeFactory(AbstractFactory):
    """
    This is a special factory for torch.optim.Optimizer.
    This factory does not instantiate an object but rather return the type, since optimizer instantiation
    requires the model to be instantiated first
    """

    def __init__(self):

        self.type_dict = {
            "SGD": optim.SGD,
            "Adam": optim.Adam,
            "RMSprop": optim.RMSprop,
            "RMSpropTF": RMSpropTF,
            "Lamb": Lamb
        }

    def get(self, conf: Union[str]):
        """
         Get a type.
            :param conf: a configuration
            if string - assumed to be a type name (not the real name, but a name defined in the Factory)
            a dictionary is not supported, since the actual instantiation takes place elsewhere

            If provided value is not one of the three above, the value will be returned as is
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
