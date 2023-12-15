from typing import Union, Mapping

from omegaconf import ListConfig, DictConfig, OmegaConf

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
    REQUIRED_BBOX_PARAM_KEYS = ["min_area", "min_visibility", "min_width", "min_height", "check_each_transform"]
    FIXED_BBOX_PARAMS = {"format": "pascal_voc", "label_fields": ["labels", "is_crowd"]}

    def __init__(self):
        if imported_albumentations_failure:
            raise imported_albumentations_failure
        super().__init__(ALBUMENTATIONS_TRANSFORMS)

    def get(self, conf: Union[str, dict]):
        if isinstance(conf, Mapping):
            if isinstance(conf, DictConfig):
                conf = OmegaConf.to_container(conf, resolve=True)
            _type = list(conf.keys())[0]  # THE TYPE NAME
            if _type in ALBUMENTATIONS_COMP_TRANSFORMS:
                conf[_type]["transforms"] = ListFactory(AlbumentationsTransformsFactory()).get(conf[_type]["transforms"])
                if "bbox_params" in conf[_type].keys():
                    bbox_params = conf[_type]["bbox_params"]
                    self._check_bbox_params(bbox_params)
                    conf[_type]["bbox_params"].update(self.FIXED_BBOX_PARAMS)

        return super(AlbumentationsTransformsFactory, self).get(conf)

    def _check_bbox_params(self, bbox_params):
        # Check if all required keys are present
        missing_keys = set(self.REQUIRED_BBOX_PARAM_KEYS) - set(bbox_params.keys())
        if missing_keys:
            raise ValueError(f"Missing required bbox_params keys: {missing_keys}")

        # Check if any fixed keys are present
        fixed_keys = set(self.FIXED_BBOX_PARAMS.keys()) & set(bbox_params.keys())
        if fixed_keys:
            raise ValueError(f"Unexpected fixed bbox_params keys: {fixed_keys}. Fixed bbox_params {self.FIXED_BBOX_PARAMS} cannot be overriden.")
