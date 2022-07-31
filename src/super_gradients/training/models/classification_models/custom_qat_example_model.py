from collections import OrderedDict
import torch
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn

from super_gradients.training.models import SgModule
from super_gradients.training.utils import HpmStruct
from super_gradients.training.utils.fine_grain_quantization_utils import SkipQuantization, QuantizationUtility, \
    RegisterQuantizedModule
from super_gradients.training.utils.quantization.core_classes import SGQuantLinear, SGTensorQuantizer, \
    SGQuantInputAndWeights


class MyQATFriendlyBlock(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.linear1 = nn.Linear(n_features, n_features)
        self.relu = nn.ReLU()
        self.n_features = n_features

    def forward(self, x):
        return x + self.relu(self.linear1(x))


@RegisterQuantizedModule(float_module=MyQATFriendlyBlock, weights_quant_descriptor=QuantDescriptor(calib_method='max'))
class QuantizedMyQATFriendlyBlock(SGQuantInputAndWeights):

    def __init__(self, n_features, **kwargs) -> None:
        # kwargs WILL INCLUDE `quant_desc_input` AND `quant_desc_weight`
        super().__init__()
        self.linear1 = SGQuantLinear(n_features, n_features, **kwargs)
        self.relu = nn.ReLU()
        self.qt = SGTensorQuantizer(quant_desc=kwargs['quant_desc_input'])

    def forward(self, x):
        return self.qt(x) + self.relu(self.linear1(x))


class MyCustomBlock(nn.Module):

    def __init__(self, in_feats, out_feats) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_feats, in_feats + out_feats)
        self.linear2 = SkipQuantization(nn.Linear(in_feats + out_feats, out_feats))

    def forward(self, x):
        return self.linear2(self.linear1(x))


class CustomQATExampleModel(SgModule):

    def __init__(self, arch_params) -> None:
        super().__init__()
        self.stem1 = SkipQuantization(nn.Conv2d(3, 16, kernel_size=3))
        self.stem2 = nn.Conv2d(16, 32, kernel_size=3)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pooling', nn.AdaptiveAvgPool2d(64)),
        ]))
        self.qat_friendly = MyQATFriendlyBlock(64)
        self.classifier = MyCustomBlock(64, arch_params.num_classes)

    def forward(self, x):
        y = self.stem1(x)
        y = self.stem2(y)
        y = self.features(y)
        y = self.qat_friendly(y)
        return self.classifier(y)

    def replace_head(self, **kwargs):
        pass


def test_wrap_with_skip_quantization():
    arch_params = {
        'num_classes': 10
    }

    my_module = CustomQATExampleModel(HpmStruct(**arch_params))
    qu = QuantizationUtility()
    qu.wrap_with_skip_quantization(my_module, {'stem2', 'features.conv0', 'classifier.linear1'})
    print(my_module)


def main():
    arch_params = {
        'num_classes': 10
    }

    my_module = CustomQATExampleModel(HpmStruct(**arch_params))
    x = torch.rand(1, 3, 128, 128)
    _ = my_module(x)
    print(my_module)
    qu = QuantizationUtility()
    qu.quantize_module(my_module)
    print('*' * 20)
    print(my_module)
    _ = my_module(x)


if __name__ == '__main__':
    # test_wrap_with_skip_quantization()
    main()
