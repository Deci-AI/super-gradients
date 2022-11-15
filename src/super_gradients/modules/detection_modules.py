from typing import Union, List
from abc import abstractmethod, ABC

import torch
from torch import nn
from omegaconf import DictConfig

from super_gradients.training.utils.utils import HpmStruct
import super_gradients.common.factories.detection_modules_factory as det_factory


class BaseDetectionModule(nn.Module, ABC):
    """
    An interface for a module that is easy to integrate into a model with complex connections
    """

    def __init__(self, in_channels: Union[List[int], int], **kwargs):
        """
        :param in_channels: defines channels of tensor(s) that will be accepted by a module in forward
        """
        super().__init__()
        self.in_channels = in_channels

    @property
    @abstractmethod
    def out_channels(self) -> Union[List[int], int]:
        raise NotImplementedError()


class NStageBackbone(BaseDetectionModule):
    """
    A backbone with a stem -> N stages -> context module
    Returns outputs of the layers listed in out_layers
    """

    def __init__(
        self,
        in_channels: int,
        out_layers: List[str],
        stem: Union[str, HpmStruct, DictConfig],
        stages: Union[str, HpmStruct, DictConfig],
        context_module: Union[str, HpmStruct, DictConfig],
    ):
        """
        :param out_layers: names of layers to output from the following options: 'stem', 'stageN', 'context_module'
        """
        super().__init__(in_channels)
        factory = det_factory.DetectionModulesFactory()

        self.num_stages = len(stages)
        self.stem = factory.get(factory.insert_module_param(stem, "in_channels", in_channels))
        prev_channels = self.stem.out_channels
        for i in range(self.num_stages):
            new_stage = factory.get(factory.insert_module_param(stages[i], "in_channels", prev_channels))
            setattr(self, f"stage{i + 1}", new_stage)
            prev_channels = new_stage.out_channels
        self.context_module = factory.get(factory.get(factory.insert_module_param(context_module, "in_channels", prev_channels)))

        self.out_layers = out_layers
        self._out_channels = self._define_out_channels()

    def _define_out_channels(self):
        out_channels = []
        for layer in self.out_layers:
            out_channels.append(getattr(self, layer).out_channels)
        return out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):

        outputs = []
        all_layers = ["stem"] + [f"stage{i}" for i in range(1, self.num_stages + 1)] + ["context_module"]
        for layer in all_layers:
            x = getattr(self, layer)(x)
            if layer in self.out_layers:
                outputs.append(x)

        return outputs


class PANNeck(BaseDetectionModule):
    """
    A PAN (path aggregation network) neck with 4 stages (2 up-sampling and 2 down-sampling stages)
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
        super().__init__(in_channels)
        c3_out_channels, c4_out_channels, c5_out_channels = in_channels

        factory = det_factory.DetectionModulesFactory()
        self.neck1 = factory.get(factory.insert_module_param(neck1, "in_channels", [c5_out_channels, c4_out_channels]))
        self.neck2 = factory.get(factory.insert_module_param(neck2, "in_channels", [self.neck1.out_channels[1], c3_out_channels]))
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

    def forward(self, inputs):
        c3, c4, c5 = inputs

        x_n1_inter, x = self.neck1([c5, c4])
        x_n2_inter, p3 = self.neck2([x, c3])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])

        return p3, p4, p5


class NHeads(BaseDetectionModule):
    """
    Apply N heads in parallel and combine predictions into the shape expected by SG detection losses
    """

    def __init__(self, in_channels: List[int], num_classes: int, heads_list: Union[str, HpmStruct, DictConfig]):
        super().__init__(in_channels)
        factory = det_factory.DetectionModulesFactory()
        heads_list = self._pass_num_classes(heads_list, factory, num_classes)

        self.num_heads = len(heads_list)
        for i in range(self.num_heads):
            new_head = factory.get(factory.insert_module_param(heads_list[i], "in_channels", in_channels[i]))
            setattr(self, f"head{i + 1}", new_head)

    @staticmethod
    def _pass_num_classes(heads_list, factory, num_classes):
        for i in range(len(heads_list)):
            heads_list[i] = factory.insert_module_param(heads_list[i], "num_classes", num_classes)
        return heads_list

    @property
    def out_channels(self):
        return None

    def forward(self, inputs):
        outputs = []
        for i in range(self.num_heads):
            outputs.append(getattr(self, f"head{i + 1}")(inputs[i]))

        return self.combine_preds(outputs)

    def combine_preds(self, preds):
        outputs = []
        outputs_logits = []
        for output, output_logits in preds:
            outputs.append(output)
            outputs_logits.append(output_logits)

        return outputs if self.training else (torch.cat(outputs, 1), outputs_logits)


ALL_DETECTION_MODULES = {
    "NStageBackbone": NStageBackbone,
    "PANNeck": PANNeck,
    "NHeads": NHeads,
}
