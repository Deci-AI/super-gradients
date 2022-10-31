from typing import Union, Type, Mapping

from super_gradients.common.factories.base_factory import AbstractFactory
from super_gradients.training.utils.activations_utils import get_builtin_activation_type


class ActivationsTypeFactory(AbstractFactory):
    """
    This is a special factory for getting a type of the activation function by name.
    This factory does not instantiate a module, but rather return the type to be instantiated via call method.
    """

    def get(self, conf: Union[str, Mapping]) -> Type:
        """
        Get a type.
           :param conf: a configuration
           if string - assumed to be a type name (not the real name, but a name defined in the Factory)
           a dictionary is not supported, since the actual instantiation takes place elsewhere

           If provided value is not one of the three above, the value will be returned as is
        """
        if isinstance(conf, str):
            return get_builtin_activation_type(conf)

        if isinstance(conf, Mapping):
            (type_name,) = list(conf.keys())
            type_args = conf[type_name]
            return get_builtin_activation_type(type_name, **type_args)

        raise RuntimeError(f"Unsupported conf param type {type(conf)}")
