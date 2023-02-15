from super_gradients.training.models.sg_module import SgModule
import torch.nn as nn
from abc import abstractmethod, ABC
from typing import Union


class SegmentationModule(SgModule, ABC):
    """
    Base SegmentationModule class
    """

    def __init__(self, use_aux_heads: bool):
        super().__init__()
        self._use_aux_heads = use_aux_heads

    @property
    def use_aux_heads(self):
        return self._use_aux_heads

    @use_aux_heads.setter
    def use_aux_heads(self, use_aux: bool):
        """
        public setter for self._use_aux_heads, called every time an assignment to self.use_aux_heads is applied.
        if use_aux is False, `_remove_auxiliary_heads` is called to delete auxiliary and detail heads.
        if use_aux is True, and self._use_aux_heads was already set to False a ValueError is raised, recreating
            aux and detail heads outside init method is not allowed, and the module should be recreated.
        """
        if use_aux is True and self._use_aux_heads is False:
            raise ValueError(
                "Cant turn use_aux_heads from False to True. Try initiating the module again with"
                " `use_aux_heads=True` or initiating the auxiliary heads modules manually."
            )
        if not use_aux:
            self._remove_auxiliary_heads()
        self._use_aux_heads = use_aux

    def prep_model_for_conversion(self, input_size: Union[tuple, list] = None, **kwargs):
        # set to false and delete auxiliary and detail heads modules.
        self.use_aux_heads = False

    @abstractmethod
    def _remove_auxiliary_heads(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """
        For SgTrainer load_backbone compatibility.
        """
        raise NotImplementedError()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
