import warnings
from typing import Optional
from functools import wraps


def make_function_deprecated(version: str, new_function_name: Optional[str] = None, new_module_name: Optional[str] = None):
    """
    Decorator to mark a function as deprecated, providing a warning message with the version number when it will be removed, and how to replace it.

    :param version:             Version number when the function will be removed.
    :param new_function_name:   New name of the function if it has been renamed.
    :param new_module_name:     Module where the function has been moved. If None, this will be set as the

    Example usage:
        >> @make_deprecated('4.0.0', new_function_name='new_get_local_rank', new_module_name='new.module.path')
        >> def get_local_rank():
        >>     from new_module import get_local_rank as _get_local_rank
        >>     return _get_local_rank()

        >> from deprecated_module import get_local_rank
        >> get_local_rank()
        DeprecationWarning: You are using `deprecated_module.get_local_rank` which is deprecated and will be removed in 4.0.0.
        Please update your code to import it as follows:
          [-] from deprecated_module import get_local_rank
          [+] from new.module.path import new_get_local_rank
    """

    def decorator(old_func):
        @wraps(old_func)
        def wrapper(*args, **kwargs):
            if not wrapper._warned:
                new_name = new_function_name or old_func.__name__
                new_module = new_module_name or old_func.__module__
                reason = (
                    f"You are using `{old_func.__module__}.{old_func.__name__}` which is deprecated and will be removed in {version}.\n"
                    f"Please update your code to import it as follows:\n"
                    f"  [-] from {old_func.__module__} import {old_func.__name__}\n"
                    f"  [+] from {new_module} import {new_name}\n."
                )
                warnings.warn(reason, DeprecationWarning, stacklevel=2)
                wrapper._warned = True

            return old_func(*args, **kwargs)

        # Each decorated object will have its own _warned state
        # This state ensures that the warning will appear only once, to avoid polluting the console in case the function is called too often.
        wrapper._warned = False
        return wrapper

    return decorator
