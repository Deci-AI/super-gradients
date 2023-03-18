from typing import Union, Mapping

from omegaconf import ListConfig

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.registry.registry import TRANSFORMS
from super_gradients.common.registry.albumentation import ALBUMENTATIONS_TRANSFORMS, ALBUMENTATIONS_COMP_TRANSFORMS
from super_gradients.common.registry.albumentation import imported_albumentations_failure
from super_gradients.training.transforms.pipeline_adaptors import AlbumentationsAdaptor


class TransformsFactory(BaseFactory):
    def __init__(self):
        super().__init__(TRANSFORMS)

    def get(self, conf: Union[str, dict]):

        # SPECIAL HANDLING FOR COMPOSE AND ALBUMENTATIONS
        if isinstance(conf, Mapping) and "Albumentations" in conf:
            return AlbumentationsAdaptor(AlbumentationsTransformsFactory().get(conf["Albumentations"]))
        if isinstance(conf, Mapping) and "Compose" in conf:
            conf["Compose"]["transforms"] = ListFactory(TransformsFactory()).get(conf["Compose"]["transforms"])
        elif isinstance(conf, (list, ListConfig)):
            conf = ListFactory(TransformsFactory()).get(conf)

        return super().get(conf)


class AlbumentationsTransformsFactory(BaseFactory):
    def __init__(self):
        if imported_albumentations_failure:
            raise imported_albumentations_failure
        super().__init__(ALBUMENTATIONS_TRANSFORMS)

    def get(self, conf: Union[str, dict]):
        if isinstance(conf, Mapping):
            _type = list(conf.keys())[0]  # THE TYPE NAME
            if _type in ALBUMENTATIONS_COMP_TRANSFORMS:
                conf[_type]["transforms"] = ListFactory(AlbumentationsTransformsFactory()).get(conf[_type]["transforms"])
        return super(AlbumentationsTransformsFactory, self).get(conf)
