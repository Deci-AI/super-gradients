import nncf
import nncf.torch
import openvino as ov
import torch
from super_gradients.common.object_names import Models
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches
from super_gradients.training import models


class DummyDataset:
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.randn(1, 3, 640, 640)


def main():
    # m = MyModel(reg_max=16).eval()
    m = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").eval()
    fuse_repvgg_blocks_residual_branches(m)

    m = ov.convert_model(m, example_input=torch.randn(8, 3, 640, 640))
    # m = nn.Sequential(m.backbone, m.neck)
    calibration_dataset = nncf.Dataset(DummyDataset())
    # qm = nncf.quantize(m, calibration_dataset)

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
