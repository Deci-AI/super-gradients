from ctypes import Union
from typing import Mapping

import inspect

from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.models import ResNet18, ResNet152, ResNet101, ResNet50_3343, ResNet50, ResNet34, MobileNetV2, EfficientNet, GoogLeNet, MobileNet
from super_gradients.training.models.classification_models.regnet import AnyNetX, RegNetX
from super_gradients.training.models.classification_models.repvgg import RepVGG
from super_gradients.training.models.detection_models.csp_darknet53 import CSPDarknet53
from super_gradients.training.models.detection_models.darknet53 import Darknet53


class BackboneFactory(BaseFactory):

    def __init__(self):

        type_dict = {
            'resnet18': ResNet18,
            'resnet34': ResNet34,
            'resnet50': ResNet50,
            'resnet50_3343': ResNet50_3343,
            'resnet101': ResNet101,
            'resnet152': ResNet152,
            'mobile_net': MobileNet,
            'mobile_net_v2': MobileNetV2,
            'anynet_x': AnyNetX,
            'regnet_x': RegNetX,
            "efficient_net": EfficientNet,
            "darknet_53": Darknet53,
            'csp_dDarknet_53': CSPDarknet53,
            "googlenet": GoogLeNet,
            'rep_vgg': RepVGG,
        }

        super().__init__(type_dict)

    def get(self, conf: Union[str, dict]):
        """
         Get an instantiated object.
         same as the basic factory with minor changes - passing backbone_mode=True either in params or in arch_params (depending on signature)

            :param conf: a configuration
            if string - assumed to be a type name (not the real name, but a name defined in the Factory)
            if dictionary - assumed to be {type_name(str): {parameters...}} (single item in dict)

            If provided value is not one of the three above, the value will be returned as is
        """
        if isinstance(conf, str):
            if conf in self.type_dict:
                _type = self.type_dict[conf]
                _kwargs = inspect.signature(_type)
                _params = {'backbone_mode':True} if 'backbone_mode' in _kwargs else {}
                return _type(**_params)
            else:
                raise RuntimeError(f"Unknown object type: {conf} in configuration. valid types are: {self.type_dict.keys()}")
        elif isinstance(conf, Mapping):
            if len(conf.keys()) > 1:
                raise RuntimeError("Malformed object definition in configuration. Expecting either a string of object type or a single entry dictionary"
                                   "{type_name(str): {parameters...}}."
                                   f"received: {conf}")

            _type = list(conf.keys())[0]  # THE TYPE NAME
            _params = list(conf.values())[0]  # A DICT CONTAINING THE PARAMETERS FOR INIT
            if _type in self.type_dict:

                _type = self.type_dict[conf]
                _kwargs = inspect.signature(_type)
                if 'backbone_mode' in _kwargs:
                    _params['backbone_mode'] = True
                elif 'arch_params' in _params:
                    _params['arch_params']['backbone_mode'] = True

                return _type(**_params)
            else:
                raise RuntimeError(f"Unknown object type: {_type} in configuration. valid types are: {self.type_dict.keys()}")
        else:
            return conf
