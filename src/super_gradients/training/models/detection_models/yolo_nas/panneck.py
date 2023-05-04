from typing import Union, List, Tuple

from omegaconf import DictConfig
from torch import Tensor

from super_gradients.common.registry import register_detection_module
from super_gradients.modules.detection_modules import BaseDetectionModule
from super_gradients.training.utils.utils import HpmStruct
import super_gradients.common.factories.detection_modules_factory as det_factory


@register_detection_module("YoloNASPANNeckWithC2")
class YoloNASPANNeckWithC2(BaseDetectionModule):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
    where the up-sampling stages include a higher resolution skip
    Returns outputs of neck stage 2, stage 3, stage 4
    """

    def __init__(
        self,
        in_channels: List[int],
        neck1: Union[str, HpmStruct, DictConfig],
        neck2: Union[str, HpmStruct, DictConfig],
        neck3: Union[str, HpmStruct, DictConfig],
        neck4: Union[str, HpmStruct, DictConfig],
    ):
        """
        Initialize the PAN neck

        :param in_channels: Input channels of the 4 feature maps from the backbone
        :param neck1: First neck stage config
        :param neck2: Second neck stage config
        :param neck3: Third neck stage config
        :param neck4: Fourth neck stage config
        """
        super().__init__(in_channels)
        c2_out_channels, c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        factory = det_factory.DetectionModulesFactory()
        self.neck1 = factory.get(factory.insert_module_param(neck1, "in_channels", [c5_out_channels, c4_out_channels, c3_out_channels]))
        self.neck2 = factory.get(factory.insert_module_param(neck2, "in_channels", [self.neck1.out_channels[1], c3_out_channels, c2_out_channels]))
        self.neck3 = factory.get(factory.insert_module_param(neck3, "in_channels", [self.neck2.out_channels[1], self.neck2.out_channels[0]]))
        self.neck4 = factory.get(factory.insert_module_param(neck4, "in_channels", [self.neck3.out_channels, self.neck1.out_channels[0]]))

        self._out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        c2, c3, c4, c5 = inputs

        x_n1_inter, x = self.neck1([c5, c4, c3])
        x_n2_inter, p3 = self.neck2([x, c3, c2])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5
