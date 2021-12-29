import torch
import torch.nn as nn
import torch.nn.functional as F


class ExportableHardswish(nn.Module):
    '''
    Export-friendly version of nn.Hardswish()
    '''

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class ExportableSiLU(nn.Module):
    """
    Export-friendly version of nn.SiLU()
    From https://github.com/ultralytics/yolov5
    """
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def fuse_conv_bn(model: nn.Module, replace_bn_with_identity: bool = False):
    """
    Fuses consecutive nn.Conv2d and nn.BatchNorm2d layers recursively inplace in all of the model
    :param replace_bn_with_identity: if set to true, bn will be replaced with identity. otherwise, bn will be removed
    :param model: the target model
    :return: the number of fuses executed
    """
    children = list(model.named_children())
    counter = 0
    for i in range(len(children) - 1):
        if isinstance(children[i][1], torch.nn.Conv2d) and isinstance(children[i + 1][1], torch.nn.BatchNorm2d):
            setattr(model, children[i][0], torch.nn.utils.fuse_conv_bn_eval(children[i][1], children[i + 1][1]))
            if replace_bn_with_identity:
                setattr(model, children[i + 1][0], nn.Identity())
            else:
                delattr(model, children[i + 1][0])
            counter += 1
    for child_name, child in children:
        counter += fuse_conv_bn(child, replace_bn_with_identity)

    return counter
