import inspect
from typing import Callable, Dict, Optional


class Registry:
    def __init__(self, items: Optional[Dict[str, Callable]] = None):
        """
        :param items: Items that will be registered on instantiation
        """
        self.items = items or {}
        self.register = create_register_decorator(self.items)


def create_register_decorator(items: Dict[str, Callable]) -> Callable:
    """
    Create a decorator that registers object of specified type (model, metric, ...)

    :param items: Items to register (maps name to object that you register)
    :return:      Register function
    """

    def register(name: Optional[str] = None) -> Callable:
        """
        Set up a register decorator.

        :param name: If specified, the decorated object will be registered with this name.
        :return:     Decorator that registers the callable.
        """

        def decorator(cls: Callable) -> Callable:
            """Register the decorated callable"""
            cls_name = name if name is not None else cls.__name__

            if cls_name in items:
                ref = items[cls_name]
                raise Exception(f"`{cls_name}` is already registered and points to `{inspect.getmodule(ref).__name__}.{ref.__name__}")

            items[cls_name] = cls
            return cls

        return decorator

    return register
