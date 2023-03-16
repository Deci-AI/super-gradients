from typing import Tuple

import torch

from super_gradients.common.registry.registry import register_detection_module
from super_gradients.training.models.segmentation_models.ddrnet import DDRNet39
from super_gradients.training.utils import HpmStruct

__all__ = ["DDRNet39Backbone"]


@register_detection_module()
class DDRNet39Backbone(DDRNet39):
    """
    A somewhat frankenstein version of the DDRNet39 model that tries to be a feature extractor module.
    """

    def __init__(self, arch_params: HpmStruct):
        super().__init__(arch_params)

        # Delete everything that is not needed for feature extraction
        del self.final_layer
        if self.use_aux_heads:
            self.use_aux_heads = False
            del self.aux_head

        if self.classification_mode:
            del self.fc
            del self.average_pool
            del self.high_to_low_fusion
            del self.layer5

        self._out_channels = (self.highres_planes * self.layer5_bottleneck_expansion,)

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self._backbone.stem(x)
        x = self._backbone.layer1(x)
        x = self._backbone.layer2(self.relu(x))

        # Repeat layer 3
        x_skip = x
        for i in range(self.layer3_repeats):
            out_layer3 = self._backbone.layer3[i](self.relu(x))
            out_layer3_skip = self.layer3_skip[i](self.relu(x_skip))

            x = out_layer3 + self.down3[i](self.relu(out_layer3_skip))
            x_skip = out_layer3_skip + self.upscale(self.compression3[i](self.relu(out_layer3)), height_output, width_output)

        out_layer4 = self._backbone.layer4(self.relu(x))
        out_layer4_skip = self.layer4_skip(self.relu(x_skip))

        x = out_layer4 + self.down4(self.relu(out_layer4_skip))
        x_skip = out_layer4_skip + self.upscale(self.compression4(self.relu(out_layer4)), height_output, width_output)

        out_layer5_skip = self.layer5_skip(self.relu(x_skip))

        x = self.upscale(self.spp(self.layer5(self.relu(x))), height_output, width_output)

        return x + out_layer5_skip

    @property
    def out_channels(self) -> Tuple[int]:
        return self._out_channels


if __name__ == "__main__":
    back = DDRNet39Backbone(HpmStruct(num_classes=1)).eval().cuda()

    x = torch.randn(1, 3, 512, 512).cuda()
    y = back(x)

    print(y.shape, y.mean(), y.std())
