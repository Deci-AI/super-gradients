from pytorch_quantization import nn as quant_nn


class use_fb_fake_quant:
    """
    Context manager object to ensure that fake quantization
    state is preserved

    >>> with use_fb_fake_quant(True):
    >>>    do_stuff()
    """

    def __init__(self, enabled: bool):
        self.use_fb_fake_quant_state = None
        self.enabled = enabled

    def __enter__(self):
        self.use_fb_fake_quant_state = quant_nn.TensorQuantizer.use_fb_fake_quant
        quant_nn.TensorQuantizer.use_fb_fake_quant = self.enabled
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        quant_nn.TensorQuantizer.use_fb_fake_quant = self.use_fb_fake_quant_state
