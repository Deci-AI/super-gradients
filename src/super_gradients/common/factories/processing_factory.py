from typing import Union, Mapping

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.registry.registry import PROCESSINGS


class ProcessingFactory(BaseFactory):
    def __init__(self):
        super().__init__(PROCESSINGS)

    def get(self, conf: Union[str, dict]):
        if isinstance(conf, Mapping) and "ComposeProcessing" in conf:
            conf["ComposeProcessing"]["processings"] = ListFactory(ProcessingFactory()).get(conf["ComposeProcessing"]["processings"])
        return super().get(conf)
