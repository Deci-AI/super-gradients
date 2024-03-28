import copy
from typing import Type, Union, Mapping, Any

import nncf
import nncf.torch
import torch
from super_gradients.common.object_names import Models
from super_gradients.modules import QARepVGGBlock
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training import models
from torch import nn


# @nncf.torch.register_module("conv")
class MyModel(nn.Module):
    def __init__(self, reg_max):
        super().__init__()
        self.reg_max = reg_max
        # self.register_buffer("proj_conv", proj, persistent=False)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DummyDataset:
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.randn(4, 3, 224, 224)


class QAPartiallyFusedVGGBlock(nn.Module):
    @classmethod
    def from_non_fused(cls, layer: QARepVGGBlock):
        kernel, bias = layer._get_equivalent_kernel_bias_for_branches()

        new_layer = cls(
            in_channels=layer.branch_3x3.conv.in_channels,
            out_channels=layer.branch_3x3.conv.out_channels,
            stride=layer.branch_3x3.conv.stride,
            dilation=layer.branch_3x3.conv.dilation,
            groups=layer.branch_3x3.conv.groups,
            activation_type=layer.activation_type,
            activation_kwargs=layer.activation_kwargs,
            use_post_bn=layer.use_post_bn,
        )

        with torch.no_grad():
            new_layer.rbr_reparam.weight.data = kernel
            new_layer.rbr_reparam.bias.data = bias

            if layer.use_post_bn:
                new_layer.post_bn.weight.data = layer.post_bn.weight
                new_layer.post_bn.bias.data = layer.post_bn.bias
                new_layer.post_bn.running_mean.data = layer.post_bn.running_mean
                new_layer.post_bn.running_var.data = layer.post_bn.running_var
                new_layer.post_bn.eps = layer.post_bn.eps
        return new_layer

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation_type: Type[nn.Module] = nn.ReLU,
        activation_kwargs: Union[Mapping[str, Any], None] = None,
        se_type: Type[nn.Module] = nn.Identity,
        se_kwargs: Union[Mapping[str, Any], None] = None,
        use_post_bn: bool = True,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param activation_type: Type of the nonlinearity (nn.ReLU by default)
        :param se_type: Type of the se block (Use nn.Identity to disable SE)
        :param stride: Output stride
        :param dilation: Dilation factor for 3x3 conv
        :param groups: Number of groups used in convolutions
        :param activation_kwargs: Additional arguments for instantiating activation module.
        :param se_kwargs: Additional arguments for instantiating SE module.
        :param build_residual_branches: Whether to initialize block with already fused parameters (for deployment)
        :param use_residual_connection: Whether to add input x to the output (Enabled in RepVGG, disabled in PP-Yolo)
        :param use_alpha: If True, enables additional learnable weighting parameter for 1x1 branch (PP-Yolo-E Plus)
        :param use_1x1_bias: If True, enables bias in the 1x1 convolution, authors don't mention it specifically
        :param use_post_bn: If True, adds BatchNorm after the sum of three branches (S4), if False, BatchNorm is not added (S3)
        """
        super().__init__()

        if activation_kwargs is None:
            activation_kwargs = {}
        if se_kwargs is None:
            se_kwargs = {}

        self.use_post_bn = use_post_bn

        self.rbr_reparam = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        if self.use_post_bn:
            self.post_bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.post_bn = nn.Identity()

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = se_type(**se_kwargs)

    def forward(self, inputs):
        return self.se(self.nonlinearity(self.post_bn(self.rbr_reparam(inputs))))


def replace_modules(model, filter_fn, replace_fn):
    for name, module in model.named_children():
        if filter_fn(module):
            setattr(model, name, replace_fn(module))
        else:
            replace_modules(module, filter_fn, replace_fn)
    return model


def do_partial_fusion(model):
    def replace_fn(layer):
        # Define your replacement layer here
        return QAPartiallyFusedVGGBlock.from_non_fused(layer)

    def filter_fn(layer):
        return isinstance(layer, QARepVGGBlock)

    model = copy.deepcopy(model)
    replace_modules(model, filter_fn, replace_fn)
    return model


def move_to_cuda(x):
    return x.cuda()


def main():
    # m = MyModel(reg_max=16).eval()
    m = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").eval()
    # m = models.get(Models.YOLO_NAS_S, num_classes=80).eval()
    fuse_repvgg_blocks_residual_branches(m)
    # m = do_partial_fusion(m).eval()

    # m = nn.Sequential(m.backbone, m.neck)
    calibration_dataset = nncf.Dataset(DummyDataset())
    # qm = nncf.quantize(m, calibration_dataset)

    #      (reg_convs): Sequential(
    #         (0): ConvBNReLU(
    #           (seq): Sequential(
    #             (conv): NNCFConv2d(
    #               128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #               (pre_ops): ModuleDict(
    #                 (0): UpdateWeight(
    #                   (op): SymmetricQuantizer(bit=8, ch=True)
    #                 )
    #               )
    #               (post_ops): ModuleDict()
    #             )
    #             (bn): NNCFBatchNorm2d(
    #               128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True
    #               (pre_ops): ModuleDict()
    #               (post_ops): ModuleDict()
    #             )
    #             (act): ReLU(inplace=True)
    #           )
    #         )
    #       )
    qm = nncf.quantize(
        m,
        calibration_dataset,
        ignored_scope=nncf.IgnoredScope(
            patterns=[
                ".+reg_convs.+",
                ".+cls_convs.+",
                ".+cls_pred.+",
                ".+reg_pred.+",
                # ".+proj_conv.+",
            ],
            # names=[
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/Sequential[reg_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFConv2d[conv]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/Sequential[reg_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFBatchNorm2d[bn]/batch_norm_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/Sequential[reg_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFConv2d[conv]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/Sequential[reg_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFBatchNorm2d[bn]/batch_norm_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/Sequential[reg_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFConv2d[conv]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/Sequential[reg_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFBatchNorm2d[bn]/batch_norm_0",
            #     #
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/Sequential[cls_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFConv2d[conv]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/Sequential[cls_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFBatchNorm2d[bn]/batch_norm_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/Sequential[cls_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFConv2d[conv]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/Sequential[cls_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFBatchNorm2d[bn]/batch_norm_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/Sequential[cls_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFConv2d[conv]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/Sequential[cls_convs]/ConvBNReLU[0]/Sequential[seq]/NNCFBatchNorm2d[bn]/batch_norm_0",
            #     #
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/NNCFConv2d[cls_pred]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/NNCFConv2d[cls_pred]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/NNCFConv2d[cls_pred]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head1]/NNCFConv2d[reg_pred]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head2]/NNCFConv2d[reg_pred]/conv2d_0",
            #     "YoloNAS_S/NDFLHeads[heads]/YoloNASDFLHead[head3]/NNCFConv2d[reg_pred]/conv2d_0",
            # ]
        ),
        subset_size=100,
    )
    print(qm)
    input = torch.randn(1, 3, 224, 224)

    output1 = m(input)  # noqa
    output2 = qm(input)  # noqa
    # print(torch.nn.functional.l1_loss(output1, output2))


if __name__ == "__main__":
    main()
