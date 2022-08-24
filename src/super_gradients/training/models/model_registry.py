import inspect
from typing import Callable, Optional

from super_gradients.training.models.all_architectures import ARCHITECTURES


def register(name: Optional[str] = None) -> Callable:
    def model_decorator(model: Callable) -> Callable:
        if name is not None:
            model_name = name
        else:
            model_name = model.__name__

        if model_name in ARCHITECTURES:
            ref = ARCHITECTURES[model_name]
            raise Exception(
                f"`{model_name}` is already registered and points to `{inspect.getmodule(ref).__name__}.{ref.__name__}"
            )

        ARCHITECTURES[model_name] = model

        return model

    return model_decorator
