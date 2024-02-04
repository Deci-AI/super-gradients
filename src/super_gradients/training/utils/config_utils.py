import abc
import os
import re
from copy import deepcopy
from collections import defaultdict
from typing import Mapping, Iterable, Set, Union, MutableMapping, MutableSequence

__all__ = ["raise_if_unused_params", "warn_if_unused_params", "check_if_unused_params", "UnusedConfigParamException"]

from omegaconf import ListConfig, DictConfig

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils import HpmStruct

logger = get_logger(__name__)


class UnusedConfigParamException(Exception):
    pass


class AccessCounterMixin:
    """
    Implements access counting mechanism for configuration settings (dicts/lists).
    It is achieved by wrapping underlying config and override __getitem__, __getattr__ methods to catch read operations
    and increments access counter for each property.
    """

    _access_counter: Mapping[str, int]
    _prefix: str  # Prefix string
    _ignore_patterns = {
        r"\.\d+$",  # ignore lists of primitives - usually the list itself is consumed immediately, but the primitives are consumed only during training
        "_convert_",
    }

    def __init__(self, config: dict, access_counter: Mapping[str, int] = None, prefix: str = "", ignore_patterns: Set[str] = set()):
        """
        :param ignore_patterns: Patterns to ignore when counting usage. Default is empty set.
        """

        self.config = config
        self._access_counter = access_counter or defaultdict(int)
        self._prefix = str(prefix)

        self._ignore_patterns.update(ignore_patterns)

    def maybe_wrap_as_counter(self, value, key, count_usage: bool = True):
        """
        Return an attribute value optionally wrapped as access counter adapter to trace read counts.

        :param value: Attribute value
        :param key: Attribute name
        :param count_usage: Whether increment usage count for given attribute. Default is True.

        :return: wrapped value
        """
        key_with_prefix = self._prefix + str(key)
        if count_usage:
            self._access_counter[key_with_prefix] += 1
        if isinstance(value, (Mapping, MutableMapping, DictConfig)):
            return AccessCounterDict(value, access_counter=self._access_counter, prefix=key_with_prefix + ".")
        if isinstance(value, (Iterable, MutableSequence, ListConfig)) and not isinstance(value, str):
            return AccessCounterList(value, access_counter=self._access_counter, prefix=key_with_prefix + ".")
        return value

    @property
    def access_counter(self):
        return self._access_counter

    @abc.abstractmethod
    def get_all_params(self) -> Set[str]:
        raise NotImplementedError()

    def get_used_params(self) -> Set[str]:
        used_params = {k for (k, v) in self._access_counter.items() if v > 0}
        return used_params

    def get_unused_params(self) -> Set[str]:
        unused_params = self._filter_by_pattern(self.get_all_params() - self.get_used_params())
        return unused_params

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _filter_by_pattern(self, param_names: Set[str]) -> Set[str]:
        out_list = []
        for param_name in param_names:
            match = False
            for pattern in self._ignore_patterns:
                if re.search(pattern, param_name):
                    match = True
                    break
            if not match:
                out_list.append(param_name)
        return out_list


class AccessCounterDict(AccessCounterMixin, Mapping):
    def __init__(self, config: dict, access_counter: Mapping[str, int] = None, prefix: str = "", ignore_patterns: Set[str] = set()):
        super().__init__(config, access_counter, prefix, ignore_patterns)

    def __iter__(self):
        return self.config.__iter__()

    def __len__(self):
        return self.config.__len__()

    def __getitem__(self, item):
        return self.get(item)

    def get_first_value(self):
        return self.maybe_wrap_as_counter(list(self.config.values())[0], str(list(self.config.keys())[0]))

    def keys(self):
        return self.config.keys()

    def __getattr__(self, item):
        if item in self.config:
            value = self.config.__getitem__(item)
            return self.maybe_wrap_as_counter(value, item)
        else:
            return None

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __repr__(self):
        return self.config.__repr__()

    def __str__(self):
        return self.config.__str__()

    def get(self, item, default=None):
        value = self.config.get(item, default)
        return self.maybe_wrap_as_counter(value, item)

    def get_all_params(self) -> Set[str]:
        keys = []
        for key, value in self.config.items():
            keys.append(self._prefix + str(key))
            value = self.maybe_wrap_as_counter(value, key, count_usage=False)
            if isinstance(value, AccessCounterMixin):
                keys += value.get_all_params()
        return set(keys)


class AccessCounterHpmStruct(AccessCounterMixin, Mapping):
    def __init__(self, config: HpmStruct, access_counter: Mapping[str, int] = None, prefix: str = "", ignore_patterns: Set[str] = set()):
        super().__init__(config, access_counter, prefix, ignore_patterns)

    def __iter__(self):
        return self.config.__dict__.__iter__()

    def __len__(self):
        return self.config.__dict__.__len__()

    def __repr__(self):
        return self.config.__repr__()

    def __str__(self):
        return self.config.__str__()

    def __getitem__(self, item):
        value = self.config.__dict__[item]
        return self.maybe_wrap_as_counter(value, item)

    def __getattr__(self, item):
        value = self.config.__dict__[item]
        return self.maybe_wrap_as_counter(value, item)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return self.config.__dict__.__contains__(key)

    def get(self, item, default=None):
        value = self.config.__dict__.get(item, default)
        return self.maybe_wrap_as_counter(value, item)

    def get_all_params(self) -> Set[str]:
        keys = []
        for key, value in self.config.__dict__.items():
            # Exclude schema field from params
            if key == "schema":
                continue
            keys.append(self._prefix + str(key))
            value = self.maybe_wrap_as_counter(value, key, count_usage=False)
            if isinstance(value, AccessCounterMixin):
                keys += value.get_all_params()
        return set(keys)

    def to_dict(self, include_schema=True) -> dict:
        return self.config.to_dict(include_schema=include_schema)

    def override(self, **kwargs):
        self.config.override(**kwargs)


class AccessCounterTrainingParams(AccessCounterHpmStruct):
    def __init__(self, config: HpmStruct, access_counter: Mapping[str, int] = None, prefix: str = ""):
        super().__init__(
            config,
            access_counter=access_counter,
            prefix=prefix,
            ignore_patterns={
                # may not be in use when torch_compile_loss = False
                "torch_compile_options",
                "torch_compile_options.fullgraph",
                "torch_compile_options.dynamic",
                "torch_compile_options.disable",
                "torch_compile_options.backend",
                "torch_compile_options.mode",
                "torch_compile_options.options",
                # may not be in use when ema = false
                "ema_params",
                "ema_params.beta",
                "ema_params.decay",
                "ema_params.decay_type",
                # TODO fix lr params structure
                "lr_cooldown_epochs",
                "cosine_final_lr_ratio",
                "lr_schedule_function",
                "lr_warmup_steps",
                "step_lr_update_freq",
                "lr_updates",
            },
        )


class AccessCounterList(AccessCounterMixin, list):
    def __init__(self, config: Iterable, access_counter: Mapping[str, int] = None, prefix: str = "", ignore_patterns: Set[str] = set()):
        super().__init__(config, access_counter, prefix, ignore_patterns)

    def __iter__(self):
        for index, value in enumerate(super().__iter__()):
            yield self.maybe_wrap_as_counter(value, index)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return self.maybe_wrap_as_counter(value, item)

    def get_all_params(self) -> Set[str]:
        keys = []
        for index, value in enumerate(super().__iter__()):
            keys.append(self._prefix + str(index))
            value = self.maybe_wrap_as_counter(value, index, count_usage=False)
            if isinstance(value, AccessCounterMixin):
                keys += value.get_all_params()
        return set(keys)


class ConfigInspector:
    def __init__(self, wrapped_config, unused_params_action: str):
        self.wrapped_config = wrapped_config
        self.unused_params_action = unused_params_action

    def __enter__(self):
        return self.wrapped_config

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise
        unused_params = self.wrapped_config.get_unused_params()
        if len(unused_params):
            unused_params = "\n\t".join(sorted(unused_params))
            message = f"Detected unused parameters in configuration object that were not consumed by caller: \n\t{unused_params}"
            if self.unused_params_action == "raise":
                raise UnusedConfigParamException(message)
            elif self.unused_params_action == "warn":
                logger.warning(message)
            elif self.unused_params_action == "ignore":
                pass
            else:
                raise KeyError(f"Encountered unknown action key {self.unused_params_action}")


def raise_if_unused_params(config: Union[HpmStruct, DictConfig, ListConfig, Mapping, list, tuple]) -> ConfigInspector:
    """
    A helper function to check whether all confuration parameters were used on given block of code. Motivation to have
    this check is to ensure there were no typo or outdated configuration parameters.
    It at least one of config parameters was not used, this function will raise an UnusedConfigParamException exception.
    Example usage:

    >>> from super_gradients.training.utils import raise_if_unused_params
    >>>
    >>> with raise_if_unused_params(some_config) as some_config:
    >>>    do_something_with_config(some_config)
    >>>

    :param config: A config to check
    :return: An instance of ConfigInspector
    """
    wrapper_cls = _wrap_class(config)

    return ConfigInspector(wrapper_cls(config), unused_params_action="raise")


def _wrap_class(config):
    if isinstance(config, TrainingParams):
        wrapper_cls = AccessCounterTrainingParams
    elif isinstance(config, HpmStruct):
        wrapper_cls = AccessCounterHpmStruct
    elif isinstance(config, (Mapping, DictConfig)):
        wrapper_cls = AccessCounterDict
    elif isinstance(config, (list, tuple, ListConfig)):
        wrapper_cls = AccessCounterList
    else:
        raise RuntimeError(f"Unsupported type. Root configuration object must be a mapping or list. Got type {type(config)}")
    return wrapper_cls


def warn_if_unused_params(config):
    """
    A helper function to check whether all confuration parameters were used on given block of code. Motivation to have
    this check is to ensure there were no typo or outdated configuration parameters.
    It at least one of config parameters was not used, this function will emit warning.
    Example usage:

    >>> from super_gradients.training.utils import warn_if_unused_params
    >>>
    >>> with warn_if_unused_params(some_config) as some_config:
    >>>    do_something_with_config(some_config)
    >>>

    :param config: A config to check
    :return: An instance of ConfigInspector
    """
    wrapper_cls = _wrap_class(config)

    return ConfigInspector(wrapper_cls(config), unused_params_action="warn")


def check_if_unused_params(config):
    """
    A helper function to check whether all confuration parameters were used on given block of code. Motivation to have
    this check is to ensure there were no typo or outdated configuration parameters.
    It at least one of config parameters was not used, this function will emit either emit warning or raise an error (default behaviour) based
    on the value of UNUSED_PARAMS_ACTION environment variable ('warn', 'raise', 'ignore').
    Example usage:

    >>> from super_gradients.training.utils import warn_if_unused_params
    >>>
    >>> with check_if_unused_params(some_config) as some_config:
    >>>    do_something_with_config(some_config)
    >>>

    :param config: A config to check
    :return: An instance of ConfigInspector
    """
    wrapper_cls = _wrap_class(config)
    variable_value = os.environ.get("UNUSED_PARAMS_ACTION", "raise")

    return ConfigInspector(wrapper_cls(config), unused_params_action=variable_value)
