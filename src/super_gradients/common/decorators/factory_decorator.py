import inspect
from functools import wraps

from super_gradients.common.factories.base_factory import AbstractFactory


def _assign_tuple(t: tuple, index: int, value):
    return tuple([x if i != index else value for i, x in enumerate(t)])


def resolve_param(param_name: str, factory: AbstractFactory):

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if param_name in kwargs:
                # handle kwargs
                kwargs[param_name] = factory.get(kwargs[param_name])
            else:
                # handle args
                func_args = inspect.getfullargspec(func).args
                if param_name in func_args:
                    index = func_args.index(param_name)
                    new_value = factory.get(args[index])
                    args = _assign_tuple(args, index, new_value)
            return func(*args, **kwargs)
        return wrapper
    return inner
