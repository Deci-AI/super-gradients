import abc
from collections import defaultdict
from typing import Mapping, Iterable, Set

__all__ = ["raise_if_unused_params", "warn_if_unused_params"]

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


class AccessCounterMixin:
    _access_counter: Mapping[str, int]
    _prefix: str

    def maybe_wrap_as_counter(self, value, key, count_usage: bool = True):
        key_with_prefix = self._prefix + str(key)
        if count_usage:
            self._access_counter[key_with_prefix] += 1
        if isinstance(value, Mapping):
            return AccessCounterDict(value, access_counter=self._access_counter, prefix=key_with_prefix + ".")
        if isinstance(value, Iterable) and not isinstance(value, str):
            return AccessCounterList(value, access_counter=self._access_counter, prefix=key_with_prefix + ".")
        return value

    @property
    def access_counter(self):
        return self._access_counter

    @abc.abstractmethod
    def get_all_properties(self) -> Set[str]:
        raise NotImplementedError()

    @property
    def all_params(self) -> Set[str]:
        return self.get_all_properties()

    @property
    def unused_params(self) -> Set[str]:
        used_props = {k for (k, v) in self._access_counter.items() if v > 0}
        unsued_props = self.all_params - used_props
        return unsued_props


class AccessCounterDict(Mapping, AccessCounterMixin):
    def __init__(self, config: Mapping, access_counter=None, prefix=""):
        super().__init__()
        self.config = config
        self._access_counter = access_counter or defaultdict(int)
        self._prefix = str(prefix)

    def __iter__(self):
        return self.config.__iter__()

    def __len__(self):
        return self.config.__len__()

    def __getitem__(self, item):
        return self.get(item)

    def __repr__(self):
        return self.config.__repr__()

    def __str__(self):
        return self.config.__str__()

    def get(self, item, default=None):
        value = self.config.get(item, default)
        return self.maybe_wrap_as_counter(value, item)

    def get_all_properties(self) -> Set[str]:
        keys = []
        for key, value in self.config.items():
            keys.append(self._prefix + str(key))
            value = self.maybe_wrap_as_counter(value, key, count_usage=False)
            if isinstance(value, AccessCounterMixin):
                keys += value.get_all_properties()
        return set(keys)


class AccessCounterList(list, AccessCounterMixin):
    def __init__(self, config: Iterable, access_counter=None, prefix=""):
        super().__init__(config)
        self._access_counter = access_counter or defaultdict(int)
        self._prefix = str(prefix)

    def __iter__(self):
        for index, value in enumerate(super().__iter__()):
            yield self.maybe_wrap_as_counter(value, index)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return self.maybe_wrap_as_counter(value, item)

    def get_all_properties(self) -> Set[str]:
        keys = []
        for index, value in enumerate(super().__iter__()):
            keys.append(self._prefix + str(index))
            value = self.maybe_wrap_as_counter(value, index, count_usage=False)
            if isinstance(value, AccessCounterMixin):
                keys += value.get_all_properties()
        return set(keys)


class ConfigInspector(AccessCounterDict):
    @classmethod
    def wrap(cls, config: Mapping) -> Mapping:
        return cls(config)

    def __init__(self, config, unused_params_action: str = "raise"):
        super().__init__(config)
        self.unused_params_action = unused_params_action

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        unused_params = self.config.unused_params
        if len(unused_params):
            if self.unused_params_action == "raise":
                raise ValueError(unused_params)
            elif self.unused_params_action == "warn":
                logger.warning(f"Detected unused parameters in configuration object that were not consumed by caller: {unused_params}")


def raise_if_unused_params(config):
    return ConfigInspector(config, unused_params_action="raise")


def warn_if_unused_params(config):
    return ConfigInspector(config, unused_params_action="warn")
