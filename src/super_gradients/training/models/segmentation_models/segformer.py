import torch
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.utils import get_param
from super_gradients.training.models.segmentation_models.segmentation_module import SegmentationModule
from super_gradients.training.utils.regularization_utils import DropPath
from super_gradients.modules.conv_bn_relu_block import ConvBNReLU
from super_gradients.common.object_names import Models
from super_gradients.common.registry.registry import register_model


from typing import List, Tuple

"""
paper:  SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
        ( https://arxiv.org/pdf/2105.15203.pdf )

Code and Imagenet-1k pre-trained backbone weights adopted from GitHub repo:
https://github.com/sithu31296/semantic-segmentation
"""


# TODO: extract this block to src/super_gradients/modules/transformer_modules and reuse the same module of Beit and
#       other ViTs
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, stride: int, padding: int):
        """
        Overlapped patch merging (https://arxiv.org/pdf/2105.15203.pdf)
        :param in_channels: number of input channels
        :param out_channels: number of output channels (embedding dimension)
        :param patch_size: patch size (k for size (k, k))
        :param stride: patch stride (k for size (k, k))
        :param padding:  patch padding (k for size (k, k))
        """

        super().__init__()

        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w


# TODO: extract this block to src/super_gradients/modules/transformer_modules and reuse the same module of Beit and
#       other ViTs
class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, head: int, sr_ratio: int):
        """
        Efficient self-attention (https://arxiv.org/pdf/2105.15203.pdf)
        :param dim: embedding dimension
        :param head: number of attention heads
        :param sr_ratio: the reduction ratio of the efficient self-attention
        """

        super().__init__()

        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.head, c // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
            x = self.sr(x).reshape(b, c, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(b, -1, 2, self.head, c // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, in_dim: int, inter_dim: int):
        """
        MixFFN block (https://arxiv.org/pdf/2105.15203.pdf)
        :param in_dim: input dimension
        :param inter_dim: intermediate dimension
        """

        super().__init__()

        self.fc1 = nn.Linear(in_dim, inter_dim)
        self.dwconv = nn.Conv2d(in_channels=inter_dim, out_channels=inter_dim, kernel_size=3, stride=1, padding=1, groups=inter_dim)
        self.fc2 = nn.Linear(inter_dim, in_dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)

        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.fc2(F.gelu(x))

        return x


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, head: int, sr_ratio: int, dpr: float):
        """
        A single encoder block (https://arxiv.org/pdf/2105.15203.pdf)
        :param dim: embedding dimension
        :param head: number of attention heads
        :param sr_ratio: the reduction ratio of the efficient self-attention
        :param dpr: drop-path ratio
        """

        super().__init__()

        self.attn = EfficientSelfAttention(dim, head, sr_ratio)

        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MixFFN(in_dim=dim, inter_dim=dim * 4)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))

        return x


class MiTBackBone(nn.Module):
    def __init__(
        self,
        embed_dims: List[int],
        encoder_layers: List[int],
        eff_self_att_reduction_ratio: List[int],
        eff_self_att_heads: List[int],
        overlap_patch_size: List[int],
        overlap_patch_stride: List[int],
        overlap_patch_pad: List[int],
        in_channels: int,
    ):
        """
        Mixed Transformer backbone encoder (https://arxiv.org/pdf/2105.15203.pdf)
        :param embed_dims: the patch embedding dimensions (number of output channels in each encoder stage)
        :param encoder_layers: the number of encoder layers in each encoder stage
        :param eff_self_att_reduction_ratio: the reduction ratios of the efficient self-attention in each stage
        :param eff_self_att_heads: number of efficient self-attention heads in each stage
        :param overlap_patch_size:  the patch size of the overlapping patch embedding in each stage
        :param overlap_patch_stride:  the patch stride of the overlapping patch embedding in each stage
        :param overlap_patch_pad:  the patch padding of the overlapping patch embedding in each stage
        :param in_channels:  number of input channels
        """

        super().__init__()

        if not (
            len(embed_dims)
            == len(encoder_layers)
            == len(eff_self_att_reduction_ratio)
            == len(eff_self_att_heads)
            == len(overlap_patch_size)
            == len(overlap_patch_stride)
            == len(overlap_patch_pad)
        ):
            raise ValueError("All backbone hyper-parameters should be lists of the same length")

        # Patch embeddings
        self.patch_embed = []
        for stage_num in range(len(embed_dims)):
            self.patch_embed.append(
                PatchEmbedding(
                    in_channels=in_channels if stage_num == 0 else embed_dims[stage_num - 1],
                    out_channels=embed_dims[stage_num],
                    patch_size=overlap_patch_size[stage_num],
                    stride=overlap_patch_stride[stage_num],
                    padding=overlap_patch_pad[stage_num],
                )
            )
            self.add_module(f"patch_embed{stage_num+1}", self.patch_embed[stage_num])

        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_layers))]

        self.blocks = []
        self.norms = []

        layer_idx = 0
        for stage_num in range(len(embed_dims)):
            self.blocks.append(
                nn.ModuleList(
                    [
                        EncoderBlock(
                            dim=embed_dims[stage_num],
                            head=eff_self_att_heads[stage_num],
                            sr_ratio=eff_self_att_reduction_ratio[stage_num],
                            dpr=dpr[layer_idx + i],
                        )
                        for i in range(encoder_layers[stage_num])
                    ]
                )
            )
            self.norms.append(nn.LayerNorm(embed_dims[stage_num]))

            self.add_module(f"block{stage_num + 1}", self.blocks[stage_num])
            self.add_module(f"norm{stage_num + 1}", self.norms[stage_num])

            layer_idx += encoder_layers[stage_num]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b_size = x.shape[0]

        features = []
        for stage_num in range(len(self.patch_embed)):
            x, h, w = self.patch_embed[stage_num](x)

            for enc_block in self.blocks[stage_num]:
                x = enc_block(x, h, w)
            x = self.norms[stage_num](x)
            x = x.reshape(b_size, h, w, -1).permute(0, 3, 1, 2)

            features.append(x)

        return features


# TODO: extract this block to src/super_gradients/modules/transformer_modules and reuse the same module of Beit and
#       other ViTs
class MLP(nn.Module):
    def __init__(self, dim: int, embed_dim: int):
        """
        A single Linear layer, with shape pre-processing
        :param dim: input dimension
        :param embed_dim: output dimension
        """

        super().__init__()

        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        return x


class SegFormerHead(nn.Module):
    def __init__(self, encoder_dims: List[int], embed_dim: int, num_classes: int):
        """
        SegFormer decoder head (https://arxiv.org/pdf/2105.15203.pdf)
        :param encoder_dims: list of encoder embedding dimensions
        :param embed_dim: unified embedding dimension
        :param num_classes: number of predicted classes
        """
        super().__init__()

        self.linear_layers = []
        for idx, dim in enumerate(encoder_dims):
            self.linear_layers.append(MLP(dim, embed_dim))
            self.add_module(f"linear_c{idx + 1}", self.linear_layers[idx])

        self.linear_fuse = ConvBNReLU(in_channels=embed_dim * len(encoder_dims), out_channels=embed_dim, kernel_size=1, bias=False, inplace=True)
        self.linear_pred = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        b, _, h, w = features[0].shape

        out_lst = [self.linear_layers[0](features[0]).permute(0, 2, 1).reshape(b, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            out = self.linear_layers[i + 1](feature).permute(0, 2, 1).reshape(b, -1, *feature.shape[-2:])
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            out_lst.append(out)

        out = self.linear_fuse(torch.cat(out_lst[::-1], dim=1))
        out = self.linear_pred(self.dropout(out))

        return out


# TODO: add support for aux heads? (not in original impl) (currently not using)
class SegFormer(SegmentationModule):
    def __init__(
        self,
        num_classes: int,
        encoder_embed_dims: List[int],
        encoder_layers: List[int],
        eff_self_att_reduction_ratio: List[int],
        eff_self_att_heads: List[int],
        decoder_embed_dim: int,
        overlap_patch_size: List[int],
        overlap_patch_stride: List[int],
        overlap_patch_pad: List[int],
        in_channels: int = 3,
    ):
        """
        :param num_classes: number of classes
        :param encoder_embed_dims: the patch embedding dimensions (number of output channels in each encoder stage)
        :param encoder_layers: the number of encoder layers in each encoder stage
        :param eff_self_att_reduction_ratio: the reduction ratios of the efficient self-attention in each stage
        :param eff_self_att_heads: number of efficient self-attention heads in each stage
        :param overlap_patch_size:  the patch size of the overlapping patch embedding in each stage
        :param overlap_patch_stride:  the patch stride of the overlapping patch embedding in each stage
        :param overlap_patch_pad:  the patch padding of the overlapping patch embedding in each stage
        :param in_channels:  number of input channels
        """

        super().__init__(use_aux_heads=False)

        self.encoder_embed_dims = encoder_embed_dims

        self._backbone = MiTBackBone(
            embed_dims=encoder_embed_dims,
            encoder_layers=encoder_layers,
            eff_self_att_reduction_ratio=eff_self_att_reduction_ratio,
            eff_self_att_heads=eff_self_att_heads,
            overlap_patch_size=overlap_patch_size,
            overlap_patch_stride=overlap_patch_stride,
            overlap_patch_pad=overlap_patch_pad,
            in_channels=in_channels,
        )

        self.decode_head = SegFormerHead(encoder_dims=encoder_embed_dims, embed_dim=decoder_embed_dim, num_classes=num_classes)

        self.init_params()

    def init_params(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def backbone(self):
        return self._backbone

    def _remove_auxiliary_heads(self):
        pass

    def replace_head(self, new_num_classes: int, new_decoder_embed_dim: int):
        self.decode_head = SegFormerHead(encoder_dims=self.encoder_embed_dims, embed_dim=new_decoder_embed_dim, num_classes=new_num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._backbone(x)
        out = self.decode_head(features)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """
        Custom param groups for training:
        - Different lr for backbone and the rest, if `multiply_head_lr` key is in `training_params`.
        """
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        multiply_lr_params, no_multiply_params = self._separate_lr_multiply_params()
        param_groups = [
            {"named_params": no_multiply_params, "lr": lr, "name": "no_multiply_params"},
            {"named_params": multiply_lr_params, "lr": lr * multiply_head_lr, "name": "multiply_lr_params"},
        ]
        return param_groups

    def update_param_groups(self, param_groups: list, lr: float, epoch: int, iter: int, training_params: HpmStruct, total_batch: int) -> list:
        multiply_head_lr = get_param(training_params, "multiply_head_lr", 1)
        for param_group in param_groups:
            param_group["lr"] = lr
            if param_group["name"] == "multiply_lr_params":
                param_group["lr"] *= multiply_head_lr
        return param_groups

    def _separate_lr_multiply_params(self):
        """
        Separate backbone params from the rest.
        :return: iterators of groups named_parameters.
        """
        backbone_names = [n for n, p in self.backbone.named_parameters()]
        multiply_lr_params, no_multiply_params = {}, {}
        for name, param in self.named_parameters():
            if name in backbone_names:
                no_multiply_params[name] = param
            else:
                multiply_lr_params[name] = param
        return multiply_lr_params.items(), no_multiply_params.items()


class SegFormerCustom(SegFormer):
    def __init__(self, arch_params: HpmStruct):
        """
        Parse arch_params and translate the parameters to build the SegFormer architecture
        :param arch_params: architecture parameters
        """

        super().__init__(
            num_classes=arch_params.num_classes,
            encoder_embed_dims=arch_params.encoder_embed_dims,
            encoder_layers=arch_params.encoder_layers,
            eff_self_att_reduction_ratio=arch_params.eff_self_att_reduction_ratio,
            eff_self_att_heads=arch_params.eff_self_att_heads,
            decoder_embed_dim=arch_params.decoder_embed_dim,
            overlap_patch_size=arch_params.overlap_patch_size,
            overlap_patch_stride=arch_params.overlap_patch_stride,
            overlap_patch_pad=arch_params.overlap_patch_pad,
            in_channels=arch_params.in_channels,
        )


DEFAULT_SEGFORMER_PARAMS = {
    "in_channels": 3,
    "overlap_patch_size": [7, 3, 3, 3],
    "overlap_patch_stride": [4, 2, 2, 2],
    "overlap_patch_pad": [3, 1, 1, 1],
    "eff_self_att_reduction_ratio": [8, 4, 2, 1],
    "eff_self_att_heads": [1, 2, 5, 8],
}

DEFAULT_SEGFORMER_B0_PARAMS = {**DEFAULT_SEGFORMER_PARAMS, "encoder_embed_dims": [32, 64, 160, 256], "encoder_layers": [2, 2, 2, 2], "decoder_embed_dim": 256}

DEFAULT_SEGFORMER_B1_PARAMS = {
    **DEFAULT_SEGFORMER_B0_PARAMS,
    "encoder_embed_dims": [64, 128, 320, 512],
}

DEFAULT_SEGFORMER_B2_PARAMS = {**DEFAULT_SEGFORMER_B1_PARAMS, "encoder_layers": [3, 4, 6, 3], "decoder_embed_dim": 768}

DEFAULT_SEGFORMER_B3_PARAMS = {
    **DEFAULT_SEGFORMER_B2_PARAMS,
    "encoder_layers": [3, 4, 18, 3],
}

DEFAULT_SEGFORMER_B4_PARAMS = {
    **DEFAULT_SEGFORMER_B2_PARAMS,
    "encoder_layers": [3, 8, 27, 3],
}

DEFAULT_SEGFORMER_B5_PARAMS = {
    **DEFAULT_SEGFORMER_B2_PARAMS,
    "encoder_layers": [3, 6, 40, 3],
}


@register_model(Models.SEGFORMER_B0)
class SegFormerB0(SegFormerCustom):
    def __init__(self, arch_params: HpmStruct):
        """
        SegFormer B0 architecture
        :param arch_params: architecture parameters
        """

        _arch_params = HpmStruct(**DEFAULT_SEGFORMER_B0_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        super().__init__(_arch_params)


@register_model(Models.SEGFORMER_B1)
class SegFormerB1(SegFormerCustom):
    def __init__(self, arch_params: HpmStruct):
        """
        SegFormer B1 architecture
        :param arch_params: architecture parameters
        """

        _arch_params = HpmStruct(**DEFAULT_SEGFORMER_B1_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        super().__init__(_arch_params)


@register_model(Models.SEGFORMER_B2)
class SegFormerB2(SegFormerCustom):
    def __init__(self, arch_params: HpmStruct):
        """
        SegFormer B2 architecture
        :param arch_params: architecture parameters
        """

        _arch_params = HpmStruct(**DEFAULT_SEGFORMER_B2_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        super().__init__(_arch_params)


@register_model(Models.SEGFORMER_B3)
class SegFormerB3(SegFormerCustom):
    def __init__(self, arch_params: HpmStruct):
        """
        SegFormer B3 architecture
        :param arch_params: architecture parameters
        """

        _arch_params = HpmStruct(**DEFAULT_SEGFORMER_B3_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        super().__init__(_arch_params)


@register_model(Models.SEGFORMER_B4)
class SegFormerB4(SegFormerCustom):
    def __init__(self, arch_params: HpmStruct):
        """
        SegFormer B4 architecture
        :param arch_params: architecture parameters
        """

        _arch_params = HpmStruct(**DEFAULT_SEGFORMER_B4_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        super().__init__(_arch_params)


@register_model(Models.SEGFORMER_B5)
class SegFormerB5(SegFormerCustom):
    def __init__(self, arch_params: HpmStruct):
        """
        SegFormer B5 architecture
        :param arch_params: architecture parameters
        """

        _arch_params = HpmStruct(**DEFAULT_SEGFORMER_B5_PARAMS)
        _arch_params.override(**arch_params.to_dict())
        super().__init__(_arch_params)
