from typing import Callable

from super_gradients.training.models.all_architectures import ARCHITECTURES


def register(model: Callable) -> Callable:
    model_name = model.__name__
    ARCHITECTURES[model_name] = model
    return model
