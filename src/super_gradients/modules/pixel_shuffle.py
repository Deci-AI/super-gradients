import torch
import torch.nn as nn


class PixelShuffle(nn.Module):
    """
    Equivalent to nn.PixelShuffle.
    nn.PixelShuffle module is translated to `DepthToSpace` layer in ONNX, some compilation frameworks (i.e tflite),
    doesn't support this layer. In that case this module should be used, it's translated to
    reshape / transpose / reshape operations in ONNX graph.
    """

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.scale = upscale_factor

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        x = x.reshape(b, torch.div(c, self.scale * self.scale, rounding_mode="trunc"), self.scale, self.scale, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, torch.div(c, self.scale * self.scale, rounding_mode="trunc"), h * self.scale, w * self.scale)
        return x
