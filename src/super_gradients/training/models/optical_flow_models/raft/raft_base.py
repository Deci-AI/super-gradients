from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from super_gradients.module_interfaces import SupportsReplaceInputChannels


__all__ = [
    "BottleneckBlock",
    "Encoder",
    "ContextEncoder",
    "FlowHead",
    "SepConvGRU",
    "ConvGRU",
    "MotionEncoder",
    "UpdateBlock",
    "CorrBlock",
    "AlternateCorrBlock",
    "FlowIterativeBlock",
]


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class Encoder(nn.Module, SupportsReplaceInputChannels):
    def __init__(
        self,
        in_channels: int,
        in_planes: int,
        output_dim: int = 128,
        norm_fn: str = "batch",
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.norm_fn = norm_fn
        self.in_planes = in_planes

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        dim = int(self.in_planes / 32)
        self.layer1 = self._make_layer(dim * 32, stride=1)
        self.layer2 = self._make_layer((dim + 1) * 32, stride=2)
        self.layer3 = self._make_layer((dim + 2) * 32, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d((dim + 2) * 32, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

    def replace_input_channels(self, in_channels: int, compute_new_weights_fn: Optional[Callable[[nn.Module, int], nn.Module]] = None):
        from super_gradients.modules.weight_replacement_utils import replace_conv2d_input_channels

        self.conv1 = replace_conv2d_input_channels(conv=self.conv1, in_channels=in_channels, fn=compute_new_weights_fn)

    def get_input_channels(self) -> int:
        return self.conv1.in_channels


class ContextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_planes: int,
        hidden_dim: int,
        context_dim: int,
        output_dim: int = 128,
        norm_fn: str = "batch",
        dropout: float = 0.0,
    ):
        super(ContextEncoder, self).__init__()

        self.cnet = Encoder(in_channels=in_channels, in_planes=in_planes, output_dim=output_dim, norm_fn=norm_fn, dropout=dropout)

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

    def forward(self, x):
        out = self.cnet(x)
        net, inp = torch.split(out, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return net, inp


class FlowHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class MotionEncoder(nn.Module):
    def __init__(
        self,
        corr_levels: int,
        corr_radius: int,
        num_corr_conv: int,
        convc1_output_dim: int,
        convc2_output_dim: int,
        convf1_output_dim: int,
        convf2_output_dim: int,
        conv_output_dim: int,
    ):
        super(MotionEncoder, self).__init__()
        self.num_corr_conv = num_corr_conv

        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, convc1_output_dim, 1, padding=0)
        if self.num_corr_conv == 2:
            self.convc2 = nn.Conv2d(convc1_output_dim, convc2_output_dim, 3, padding=1)
            conv_input_dim = convf2_output_dim + convc2_output_dim
        else:
            conv_input_dim = convf2_output_dim + convc1_output_dim

        self.convf1 = nn.Conv2d(2, convf1_output_dim, 7, padding=3)
        self.convf2 = nn.Conv2d(convf1_output_dim, convf2_output_dim, 3, padding=1)
        self.conv = nn.Conv2d(conv_input_dim, conv_output_dim, 3, padding=1)

    def forward(self, flow: Tensor, corr: Tensor):
        cor = F.relu(self.convc1(corr))
        if self.num_corr_conv == 2:
            cor = F.relu(self.convc2(cor))

        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self, encoder_params, hidden_dim: int = 128):
        super(UpdateBlock, self).__init__()
        self.use_mask = encoder_params.update_block.use_mask

        self.encoder = MotionEncoder(
            encoder_params.corr_levels,
            encoder_params.corr_radius,
            encoder_params.update_block.motion_encoder.num_corr_conv,
            encoder_params.update_block.motion_encoder.convc1_output_dim,
            encoder_params.update_block.motion_encoder.convc2_output_dim,
            encoder_params.update_block.motion_encoder.convf1_output_dim,
            encoder_params.update_block.motion_encoder.convf2_output_dim,
            encoder_params.update_block.motion_encoder.conv_output_dim,
        )

        if encoder_params.update_block.gru.block == "ConvGRU":
            self.gru = ConvGRU(hidden_dim=encoder_params.update_block.gru.hidden_dim, input_dim=encoder_params.update_block.gru.input_dim)
        elif encoder_params.update_block.gru.block == "SepConvGRU":
            self.gru = SepConvGRU(hidden_dim=encoder_params.update_block.gru.hidden_dim, input_dim=encoder_params.update_block.gru.input_dim)

        self.flow_head = FlowHead(hidden_dim, hidden_dim=encoder_params.update_block.flow_head.hidden_dim)

        if self.use_mask:
            self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        if self.use_mask:
            # scale mask to balance gradients
            mask = 0.25 * self.mask(net)
            return net, mask, delta_flow
        else:
            return net, None, delta_flow


class CorrBlock:
    def __init__(self, num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius

    def __call__(self, coords, fmap1, fmap2):
        corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        corr_pyramid.append(corr)

        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            corr_pyramid.append(corr)

        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = self.bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

    @staticmethod
    def bilinear_sampler(img, coords, mask: bool = False):
        """Wrapper for grid_sample, uses pixel coordinates"""
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img


class AlternateCorrBlock:
    def __init__(self, num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius

    def __call__(
        self,
        coords,
        fmap1,
        fmap2,
    ):
        import alt_cuda_corr

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            (corr,) = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class FlowIterativeBlock(nn.Module):
    def __init__(self, encoder_params, hidden_dim, flow_params, alternate_corr):
        super(FlowIterativeBlock, self).__init__()
        self.update_block = UpdateBlock(encoder_params, hidden_dim)
        self.upsample_mode = flow_params.upsample_mode
        self.iters = flow_params.iters

        if alternate_corr:
            self.corr_fn = AlternateCorrBlock(radius=encoder_params.corr_radius)
        else:
            self.corr_fn = CorrBlock(radius=encoder_params.corr_radius)

    @staticmethod
    def upsample_flow(flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    @staticmethod
    def upflow8(flow, mode="bilinear"):
        new_size = (8 * flow.shape[2], 8 * flow.shape[3])
        return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def forward(self, coords0, coords1, net, inp, fmap1, fmap2):

        flow_predictions = []

        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = self.corr_fn(coords1, fmap1, fmap2)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            # update the coordinates based on the flow change
            coords1 = coords1 + delta_flow

            # upsample flow predictions
            if self.upsample_mode is None:
                flow_up = self.upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        return flow_predictions, flow_up
