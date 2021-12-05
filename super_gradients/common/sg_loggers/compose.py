from typing import Union, List

import numpy as np
from PIL import Image
import torch

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger

logger = get_logger(__name__)


class Compose(AbstractSGLogger):

    def __init__(self, sg_loggers: List[AbstractSGLogger]):
        super().__init__()
        self.sg_loggers = sg_loggers

    def add_config(self, tag: str, config: dict):
        for c in self.sg_loggers:
            c.add_config(tag=tag, config=config)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None):
        for c in self.sg_loggers:
            c.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def add_scalars(self, tag_scalar_dict: dict, global_step: int = None):
        for c in self.sg_loggers:
            c.add_scalars(tag_scalar_dict=tag_scalar_dict, global_step=global_step)

    def add_image(self, tag: str, image: Union[torch.Tensor, np.array, Image.Image], data_format='CHW', global_step: int = None):
        for c in self.sg_loggers:
            c.add_image(tag=tag, image=image, data_format=data_format, global_step=global_step)

    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.array], bins: Union[str, np.array, list, int] = 'auto', global_step: int = None):
        for c in self.sg_loggers:
            c.add_histogram(tag=tag, values=values, bins=bins, global_step=global_step)

    def add_text(self, tag: str, text_string: str, global_step: int = None):
        for c in self.sg_loggers:
            c.add_text(tag=tag, text_string=text_string, global_step=global_step)

    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = None):
        for c in self.sg_loggers:
            c.add_checkpoint(tag=tag, state_dict=state_dict, global_step=global_step)

    def upload(self):
        for c in self.sg_loggers:
            c.upload()

    def flush(self):
        for c in self.sg_loggers:
            c.flush()

    def close(self):
        for c in self.sg_loggers:
            c.close()

    def local_dir(self) -> str:
        local_dir = self.sg_loggers[0].local_dir
        for c in self.sg_loggers:
            if c.local_dir != local_dir:
                logger.warning('Composed SGLogger: sg_loggers local_dir are not the same! some files may be saved in wrong locations')

    def __contains__(self, item):
        return item in self.sg_loggers



