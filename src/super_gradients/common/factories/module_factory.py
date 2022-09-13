import inspect
import types

from super_gradients.common.factories.base_factory import BaseFactory


class ModuleFactory(BaseFactory):

    def __init__(self, module: types.ModuleType):
        super().__init__(dict(inspect.getmembers(module, inspect.isclass)))
