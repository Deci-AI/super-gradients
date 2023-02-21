from typing import List, Sequence

from super_gradients.common.factories.base_factory import AbstractFactory


class ListFactory(AbstractFactory):
    def __init__(self, factory: AbstractFactory):
        self.factory = factory

    def get(self, conf_list: List):
        if isinstance(conf_list, Sequence):
            all = []
            for conf in conf_list:
                all.append(self.factory.get(conf))
            return all
        else:
            # FALLBACK - IN CASE LIST OF TYPE OR TYPE ARE BOTH EXPECTED
            return self.factory.get(conf_list)
