from typing import Union, Tuple

from deprecated import deprecated
from omegaconf import DictConfig
from torch import nn

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.sg_trainer import Trainer

logger = get_logger(__name__)


class QATTrainer(Trainer):
    @classmethod
    @deprecated(version="3.2.0", reason="QATTrainer is deprecated and will be removed in future release, use Trainer " "class instead.")
    def quantize_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:
        return Trainer.quantize_from_config(cfg)
