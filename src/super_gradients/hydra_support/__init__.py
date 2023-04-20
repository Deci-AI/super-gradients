import functools

from .resolvers import register_hydra_resolvers
from .local_rank import pop_local_rank

__all__ = ["hydra_love"]


def hydra_love(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        pop_local_rank()
        register_hydra_resolvers()
        return function(*args, **kwargs)

    return wrapper
