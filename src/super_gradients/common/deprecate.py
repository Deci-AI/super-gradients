import warnings
from functools import wraps
from typing import Optional, Callable
from pkg_resources import parse_version


def deprecated(deprecated_since: str, removed_from: str, target: Optional[callable] = None, reason: str = ""):
    """
    Decorator to mark a callable as deprecated. Works on functions and classes.
    It provides a clear and actionable warning message informing
    the user about the version in which the function was deprecated, the version in which it will be removed,
    and guidance on how to replace it.

    :param deprecated_since: Version number when the function was deprecated.
    :param removed_from:     Version number when the function will be removed.
    :param target:           (Optional) The new function that should be used as a replacement. If provided, it will guide the user to the updated function.
    :param reason:           (Optional) Additional information or reason for the deprecation.

    Example usage:
        If a direct replacement function exists:
        >> from new.module.path import new_get_local_rank

        >> @deprecated(deprecated_since='3.2.0', removed_from='4.0.0', target=new_get_local_rank, reason="Replaced for optimization")
        >> def get_local_rank():
        >>     return new_get_local_rank()

        If there's no direct replacement:
        >> @deprecated(deprecated_since='3.2.0', removed_from='4.0.0', reason="Function is no longer needed due to XYZ reason")
        >> def some_old_function():
        >>     # ... function logic ...

        When calling a deprecated function:
        >> from some_module import get_local_rank
        >> get_local_rank()
        DeprecationWarning: Function `some_module.get_local_rank` is deprecated. Deprecated since version `3.2.0`
        and will be removed in version `4.0.0`. Reason: `Replaced for optimization`.
        Please update your code:
          [-] from `some_module` import `get_local_rank`
          [+] from `new.module.path` import `new_get_local_rank`.
    """

    def decorator(old_func: callable) -> callable:
        @wraps(old_func)
        def wrapper(*args, **kwargs):
            if not wrapper._warned:
                import super_gradients

                is_still_supported = parse_version(super_gradients.__version__) < parse_version(removed_from)
                status_msg = "is deprecated" if is_still_supported else "was deprecated and has been removed"
                message = (
                    f"Callable `{old_func.__module__}.{old_func.__name__}` {status_msg} since version `{deprecated_since}` "
                    f"and will be removed in version `{removed_from}`.\n"
                )
                if reason:
                    message += f"Reason: {reason}.\n"

                if target is not None:
                    message += (
                        f"Please update your code:\n"
                        f"  [-] from `{old_func.__module__}` import `{old_func.__name__}`\n"
                        f"  [+] from `{target.__module__}` import `{target.__name__}`"
                    )

                if is_still_supported:
                    warnings.simplefilter("once", DeprecationWarning)  # Required, otherwise the warning may never be displayed.
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                    wrapper._warned = True
                else:
                    raise ImportError(message)

            return old_func(*args, **kwargs)

        # Each decorated object will have its own _warned state
        # This state ensures that the warning will appear only once, to avoid polluting the console in case the function is called too often.
        wrapper._warned = False
        return wrapper

    return decorator


def deprecated_training_param(deprecated_tparam_name: str, deprecated_since: str, removed_from: str, new_arg_assigner: Callable, message: str = ""):
    """
    Decorator for deprecating training hyperparameters.

    Recommended tp be used as a decorator on top of super_gradients.training.params.TrainingParams's override method:

        class TrainingParams(HpmStruct):
        def __init__(self, **entries):
            # WE initialize by the default training params, overridden by the provided params
            default_training_params = deepcopy(DEFAULT_TRAINING_PARAMS)
            super().__init__(**default_training_params)
        self.set_schema(TRAINING_PARAM_SCHEMA)
            if len(entries) > 0:
                self.override(**entries)

    @deprecated_training_param(
        "criterion_params", "3.2.1", "3.3.0", new_arg_assigner=get_deprecated_nested_params_to_factory_format_assigner("loss", "criterion_params")
    )
    def override(self, **entries):
        super().override(**entries)
        self.validate()


    :param deprecated_tparam_name: str, the name of the deprecated hyperparameter.
    :param deprecated_since: str, SG version of deprecation.
    :param removed_from: str, SG version of removal.
    :param new_arg_assigner: Callable, a handler to assign the deprecated parameter value to the updated
     hyperparameter entry.
    :param message: str, message to append to the deprecation warning (default="")
    :return:
    """

    def decorator(func):
        def wrapper(*args, **training_params):
            if deprecated_tparam_name in training_params:
                import super_gradients

                is_still_supported = parse_version(super_gradients.__version__) < parse_version(removed_from)
                if is_still_supported:
                    message_prefix = (
                        f"Training hyperparameter `{deprecated_tparam_name} is deprecated since version `{deprecated_since}` "
                        f"and will be removed in version `{removed_from}`.\n"
                    )
                    warnings.warn(message_prefix + message, DeprecationWarning)
                    training_params = new_arg_assigner(**training_params)
                else:
                    message_prefix = (
                        f"Training hyperparameter `{deprecated_tparam_name} was deprecate since version `{deprecated_since}` "
                        f"and was removed in version `{removed_from}`.\n"
                    )
                    raise RuntimeError(message_prefix + message)

            return func(*args, **training_params)

        return wrapper

    return decorator


def get_deprecated_nested_params_to_factory_format_assigner(param_name: str, nested_params_name: str) -> Callable:
    """
    Returns an assigner to be used by deprecated_training_param decorator.

    The assigner takes a deprecated parameter name, and its __init___ arguments that previously were passed
     through nested_params_name entry in training_params and manipulates the training_params so they are in 'Factory' format.
     For example:

    class TrainingParams(HpmStruct):
        def __init__(self, **entries):
            # WE initialize by the default training params, overridden by the provided params
            default_training_params = deepcopy(DEFAULT_TRAINING_PARAMS)
            super().__init__(**default_training_params)
        self.set_schema(TRAINING_PARAM_SCHEMA)
            if len(entries) > 0:
                self.override(**entries)

    @deprecated_training_param(
        "criterion_params", "3.2.1", "3.3.0", new_arg_assigner=get_deprecated_nested_params_to_factory_format_assigner("loss", "criterion_params")
    )
    def override(self, **entries):
        super().override(**entries)
        self.validate()


    then under the hood, training_params.loss will be set to
     {training_params.loss: training_params.criterion_params}

    :param param_name: str, parameter name (for example, 'loss').
    :param nested_params_name: str, nested_params_name (for example, 'criterion_params')
    :return: Callable as described above.
    """

    def deprecated_nested_params_to_factory_format_assigner(**params):
        nested_params = params.get(nested_params_name)
        param_val = params.get(param_name)
        params[param_name] = {param_val: nested_params}
        return params

    return deprecated_nested_params_to_factory_format_assigner
