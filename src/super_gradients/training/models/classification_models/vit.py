"""Vision Transformer in PyTorch.
Reference:
[1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale."
arXiv preprint arXiv:2010.11929 (2020)

Code adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn
from einops import repeat

from super_gradients.common.registry.registry import register_model
from super_gradients.common.object_names import Models
from super_gradients.training.models import SgModule
from super_gradients.training.utils import get_param


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding Using Conv layers (Faster than rearranging + Linear)
    """

    def __init__(self, img_size: tuple, patch_size: tuple, in_channels=3, hidden_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(hidden_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class FeedForward(nn.Module):
    """
    feed forward block with residual connection
    """

    def __init__(self, hidden_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class Attention(nn.Module):
    """
    self attention layer with residual connection
    """

    def __init__(self, hidden_dim, heads=8):
        super().__init__()
        dim_head = hidden_dim // heads
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(hidden_dim, inner_dim * 3, bias=True)  # Qx, Kx, Vx are calculated at once
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):

        B, N, C = x.shape
        # computing query, key and value matrices at once
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, heads, mlp_dim, dropout_prob=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = Attention(hidden_dim, heads=heads)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = FeedForward(hidden_dim, mlp_dim, dropout=dropout_prob)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim, depth, heads, mlp_dim, dropout_prob=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(TransformerBlock(hidden_dim, heads, mlp_dim, dropout_prob=dropout_prob))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(SgModule):
    def __init__(
        self,
        image_size: tuple,
        patch_size: tuple,
        num_classes: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        in_channels=3,
        dropout_prob=0.0,
        emb_dropout_prob=0.0,
        backbone_mode=False,
    ):
        """
        :param image_size: Image size tuple for data processing into patches done within the model.
        :param patch_size: Patch size tuple for data processing into patches done within the model.
        :param num_classes: Number of classes for the classification head.
        :param hidden_dim: Output dimension of each transformer block.
        :param depth: Number of transformer blocks
        :param heads: Number of attention heads
        :param mlp_dim: Intermediate dimension of the transformer block's feed forward
        :param in_channels: input channels
        :param dropout: Dropout ratio between the feed forward layers.
        :param emb_dropout: Dropout ratio between after the embedding layer
        :param backbone_mode: If True output after pooling layer
        """

        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, "Image dimensions must be divisible by the patch size."
        assert hidden_dim % heads == 0, "Hidden dimension must be divisible by the number of heads."

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.patch_embedding = PatchEmbed(image_size, patch_size, in_channels=in_channels, hidden_dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.dropout = nn.Dropout(emb_dropout_prob)

        self.transformer = Transformer(hidden_dim, depth, heads, mlp_dim, dropout_prob)

        self.backbone_mode = backbone_mode
        self.pre_head_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, img):
        x = self.patch_embedding(img)  # Convert image to patches and embed
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.pre_head_norm(x)
        x = x[:, 0]
        if self.backbone_mode:
            return x
        else:
            return self.head(x)

    def replace_head(self, new_num_classes=None, new_head=None):
        if new_num_classes is None and new_head is None:
            raise ValueError("At least one of new_num_classes, new_head must be given to replace output layer.")
        if new_head is not None:
            self.head = new_head
        else:
            self.head = nn.Linear(self.head.in_features, new_num_classes)


@register_model(Models.VIT_BASE)
class ViTBase(ViT):
    def __init__(self, arch_params, num_classes=None, backbone_mode=None):
        super(ViTBase, self).__init__(
            image_size=get_param(arch_params, "image_size", (224, 224)),
            patch_size=get_param(arch_params, "patch_size", (16, 16)),
            num_classes=num_classes or arch_params.num_classes,
            hidden_dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            in_channels=get_param(arch_params, "in_channels", 3),
            dropout_prob=get_param(arch_params, "dropout_prob", 0),
            emb_dropout_prob=get_param(arch_params, "emb_dropout_prob", 0),
            backbone_mode=backbone_mode,
        )


@register_model(Models.VIT_LARGE)
class ViTLarge(ViT):
    def __init__(self, arch_params, num_classes=None, backbone_mode=None):
        super(ViTLarge, self).__init__(
            image_size=get_param(arch_params, "image_size", (224, 224)),
            patch_size=get_param(arch_params, "patch_size", (16, 16)),
            num_classes=num_classes or arch_params.num_classes,
            hidden_dim=1024,
            depth=24,
            heads=16,
            mlp_dim=4096,
            in_channels=get_param(arch_params, "in_channels", 3),
            dropout_prob=get_param(arch_params, "dropout_prob", 0),
            emb_dropout_prob=get_param(arch_params, "emb_dropout_prob", 0),
            backbone_mode=backbone_mode,
        )


@register_model(Models.VIT_HUGE)
class ViTHuge(ViT):
    def __init__(self, arch_params, num_classes=None, backbone_mode=None):
        super(ViTHuge, self).__init__(
            image_size=get_param(arch_params, "image_size", (224, 224)),
            patch_size=get_param(arch_params, "patch_size", (16, 16)),
            num_classes=num_classes or arch_params.num_classes,
            hidden_dim=1280,
            depth=32,
            heads=16,
            mlp_dim=5120,
            in_channels=get_param(arch_params, "in_channels", 3),
            dropout_prob=get_param(arch_params, "dropout_prob", 0),
            emb_dropout_prob=get_param(arch_params, "emb_dropout_prob", 0),
            backbone_mode=backbone_mode,
        )
