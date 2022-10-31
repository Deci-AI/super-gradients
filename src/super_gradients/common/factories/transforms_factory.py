from typing import Union, Mapping

from omegaconf import ListConfig

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.training.transforms import TRANSFORMS


class TransformsFactory(BaseFactory):

    def __init__(self):
        super().__init__(TRANSFORMS)

    def get(self, conf: Union[str, dict]):

        # SPECIAL HANDLING FOR COMPOSE
        if isinstance(conf, Mapping) and 'Compose' in conf:
            conf['Compose']['transforms'] = ListFactory(TransformsFactory()).get(conf['Compose']['transforms'])
        elif isinstance(conf, (list, ListConfig)):
            conf = ListFactory(TransformsFactory()).get(conf)

        return super().get(conf)
