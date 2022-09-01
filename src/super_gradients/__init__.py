from super_gradients.training import ARCHITECTURES, losses, utils, datasets_utils, DataAugmentation, \
    TestDatasetInterface, SegmentationTestDatasetInterface, DetectionTestDatasetInterface, ClassificationTestDatasetInterface, SgModel, KDModel, \
    Trainer, KDTrainer
from super_gradients.common import init_trainer, is_distributed
from super_gradients.examples.train_from_recipe_example import train_from_recipe
from super_gradients.examples.train_from_kd_recipe_example import train_from_kd_recipe
from super_gradients.sanity_check import env_sanity_check

__all__ = ['ARCHITECTURES', 'losses', 'utils', 'datasets_utils', 'DataAugmentation',
           'TestDatasetInterface', 'Trainer', 'KDTrainer', 'SegmentationTestDatasetInterface', 'DetectionTestDatasetInterface',
           'ClassificationTestDatasetInterface', 'init_trainer', 'is_distributed', 'train_from_recipe', 'train_from_kd_recipe',
           'env_sanity_check', 'KDModel', 'SgModel']


env_sanity_check()

class Model(Enum):
    ResNet18 = ("resnet18", ResNet18Cls, "resnet18_arch_params")
    ResNet34 = ("resnet34", ResNet34Cls, "resnet34_arch_params")
    ResNet50 = ("resnet50", ResNet50Cls, "resnet50_arch_params")

    def __init__(self,
                 name: str,
                 cls: Type[nn.Module],
                 default_yaml: str):
        self.name = name
        self.cls = cls
        self.default_yaml = default_yaml