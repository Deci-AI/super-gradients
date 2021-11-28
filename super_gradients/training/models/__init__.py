# flake8: noqa # FIXME: find sol for F403 error (caused by import *), most likely need to import everything by hand
from .sg_module import *
from super_gradients.training.models.classification_models.densenet import *
from super_gradients.training.models.classification_models.dpn import *
from super_gradients.training.models.classification_models.googlenet import *
from super_gradients.training.models.classification_models.lenet import *
from super_gradients.training.models.classification_models.mobilenet import *
from super_gradients.training.models.classification_models.mobilenetv2 import *
from super_gradients.training.models.classification_models.pnasnet import *
from super_gradients.training.models.classification_models.preact_resnet import *
from super_gradients.training.models.classification_models.resnet import *
from super_gradients.training.models.classification_models.resnext import *
from super_gradients.training.models.classification_models.senet import *
from super_gradients.training.models.classification_models.shufflenet import *
from super_gradients.training.models.classification_models.shufflenetv2 import *
from super_gradients.training.models.classification_models.vgg import *
from super_gradients.training.models.segmentation_models.shelfnet import *
from super_gradients.training.models.classification_models.efficientnet import *

from super_gradients.training.models.all_architectures import ARCHITECTURES
