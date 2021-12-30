"""EfficientNet model class, based on
"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`
Code source: https://github.com/lukemelas/EfficientNet-PyTorch
Pre-trained checkpoints converted to Deci's code base with the reported accuracy can be found in S3 repo
"""
#######################################################################################################################
#   1. Since each net expects a specific image size, make sure to build the dataset with the correct image size:
#         b0 - (224, 256), b1 - (240, 274), b2 - (260, 298), b3 - (300, 342), b4 - (380, 434),
#         b5 - (456, 520), b6 - (528, 602), b7 - (600, 684), b8 - (672, 768), l2 - (800, 914)
#         You should build the DataSetInterface with the following dictionary:
#           ImageNetDatasetInterface(dataset_params = {'crop': 260, 'resize':  298})
#   2. Pre-trained ImageNet models can be found in S3://deci-model-repository-research/efficientnet_b#/ckpt_best.pth
#   3. See example code in experimental/efficientnet/efficientnet_example.py
#######################################################################################################################


import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from super_gradients.training.utils import HpmStruct
from super_gradients.training.models.sg_module import SgModule

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Set BlockArgs's defaults
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth.
    Args:
        filters (int): Filters number to be calculated.
        Params from arch_params:
        width_coefficient (int): model's width coefficient. Used as the multiplier.
        depth_divisor (int): model's depth divisor. Used as the divisor.
        and min_depth (int): model's minimal depth, if given.
    Returns:
        new_filters: New filters number after calculating.
    """
    if not width_coefficient:
        return filters
    min_depth = min_depth
    filters *= width_coefficient
    min_depth = min_depth or depth_divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient.
    Args:
        repeats (int): num_repeat to be calculated.
        depth_coefficient (int): the depth coefficient of the model. this func uses it as the multiplier.
    Returns:
        new repeat: New repeat number after calculating.
    """
    if not depth_coefficient:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(depth_coefficient * repeats))


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        inputs (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert p >= 0 and p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.
    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    elif isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)

    image_height, image_width = input_image_size
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


# Note:
# The following 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names ! ! !

def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.
    Args:
        image_size (int or tuple): Size of the image.
    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w - pad_w // 2, pad_w // 2, pad_h - pad_h // 2, pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    """Identity mapping.
       Send input to output directly.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


# BlockDecoder: A Class for encoding and decoding BlockArgs
# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet

class BlockDecoder(object):
    """Block Decoder for readability, straight from the official TensorFlow repository."""

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))

    @staticmethod
    def _encode_block_string(block):
        """Encode a block to a string.
        Args:
            block (namedtuple): A BlockArgs type argument.
        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """Encode a list of BlockArgs to a list of strings.
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs.
        arch_params (HpmStruct): HpmStruct.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, batch_norm_momentum, batch_norm_epsilon, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = nn.functional.silu

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(SgModule):
    """EfficientNet model.
    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        arch_params (HpmStruct): A set of global params shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    """

    def __init__(self, blocks_args=None, arch_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._arch_params = arch_params
        self._blocks_args = blocks_args
        self.backbone_mode = arch_params.backbone_mode

        # Batch norm parameters
        bn_mom = 1 - self._arch_params.batch_norm_momentum
        bn_eps = self._arch_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = arch_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._arch_params.width_coefficient, self._arch_params.depth_divisor,
                                     self._arch_params.min_depth)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._arch_params.width_coefficient,
                                            self._arch_params.depth_divisor, self._arch_params.min_depth),
                output_filters=round_filters(block_args.output_filters, self._arch_params.width_coefficient,
                                             self._arch_params.depth_divisor, self._arch_params.min_depth),
                num_repeat=round_repeats(block_args.num_repeat, self._arch_params.depth_coefficient))

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._arch_params.batch_norm_momentum,
                                            self._arch_params.batch_norm_epsilon, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._arch_params.batch_norm_momentum,
                                                self._arch_params.batch_norm_epsilon, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._arch_params.width_coefficient, self._arch_params.depth_divisor,
                                     self._arch_params.min_depth)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        if not self.backbone_mode:
            self._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self._dropout = nn.Dropout(self._arch_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._arch_params.num_classes)
        self._swish = nn.functional.silu

    def extract_features(self, inputs):
        """
        Use convolution layer to extract feature.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution.
            layer in the efficientnet model.
        """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._arch_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        bs = inputs.size(0)

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer, not needed for backbone mode
        if not self.backbone_mode:
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)

        return x

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self._fc = new_head
        else:
            self._fc = nn.Linear(self._fc.in_features, new_num_classes)

    def load_state_dict(self, state_dict, strict=True):
        """
        load_state_dict - Overloads the base method and calls it to load a modified dict for usage as a backbone
        :param state_dict:  The state_dict to load
        :param strict:      strict loading (see super() docs)
        """
        pretrained_model_weights_dict = state_dict.copy()

        if self.backbone_mode:
            # FIRST LET'S POP THE LAST TWO LAYERS - NO NEED TO LOAD THEIR VALUES SINCE THEY ARE IRRELEVANT AS A BACKBONE
            pretrained_model_weights_dict.popitem()
            pretrained_model_weights_dict.popitem()

            pretrained_backbone_weights_dict = OrderedDict()
            for layer_name, weights in pretrained_model_weights_dict.items():
                # GET THE LAYER NAME WITHOUT THE 'module.' PREFIX
                name_without_module_prefix = layer_name.split('module.')[1]

                # MAKE SURE THESE ARE NOT THE FINAL LAYERS
                pretrained_backbone_weights_dict[name_without_module_prefix] = weights

        # RETURNING THE UNMODIFIED/MODIFIED STATE DICT DEPENDING ON THE backbone_mode VALUE
        super().load_state_dict(pretrained_backbone_weights_dict, strict)


def build_efficientnet(width, depth, res, dropout, arch_params):
    """

    :param width:
    :param depth:
    :param res:
    :param dropout:
    :param arch_params:
    :return:
    """
    print(f"\nNOTICE: \nachieving EfficientNet\'s reported accuracy requires specific image resolution."
          f"\nPlease verify image size is {res}x{res} for this specific EfficientNet configuration\n")
    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_args = BlockDecoder.decode(['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
                                       'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
                                       'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
                                       'r1_k3_s11_e6_i192_o320_se0.25', ])
    # Default values
    arch_params_new = HpmStruct(**{"width_coefficient": width, "depth_coefficient": depth, "image_size": res,
                                   "dropout_rate": dropout, "num_classes": arch_params.num_classes,
                                   "batch_norm_momentum": 0.99, "batch_norm_epsilon": 1e-3, "drop_connect_rate": 0.2,
                                   "depth_divisor": 8, "min_depth": None, "backbone_mode": False})
    # Update arch_params
    arch_params_new.override(**arch_params.to_dict())
    return EfficientNet(blocks_args, arch_params_new)


def b0(arch_params):
    return build_efficientnet(1.0, 1.0, 224, 0.2, arch_params)


def b1(arch_params):
    return build_efficientnet(1.0, 1.1, 240, 0.2, arch_params)


def b2(arch_params):
    return build_efficientnet(1.1, 1.2, 260, 0.3, arch_params)


def b3(arch_params):
    return build_efficientnet(1.2, 1.4, 300, 0.3, arch_params)


def b4(arch_params):
    return build_efficientnet(1.4, 1.8, 380, 0.4, arch_params)


def b5(arch_params):
    return build_efficientnet(1.6, 2.2, 456, 0.4, arch_params)


def b6(arch_params):
    return build_efficientnet(1.8, 2.6, 528, 0.5, arch_params)


def b7(arch_params):
    return build_efficientnet(2.0, 3.1, 600, 0.5, arch_params)


def b8(arch_params):
    return build_efficientnet(2.2, 3.6, 672, 0.5, arch_params)


def l2(arch_params):
    return build_efficientnet(4.3, 5.3, 800, 0.5, arch_params)


def CustomizedEfficientnet(arch_params):
    return build_efficientnet(arch_params.width_coefficient, arch_params.depth_coefficient, arch_params.res,
                              arch_params.dropout_rate, arch_params)
