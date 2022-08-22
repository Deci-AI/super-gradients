from super_gradients.training import utils as core_utils
from super_gradients.training.models import SgModule
from super_gradients.training.models.all_architectures import ARCHITECTURES
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model, load_pretrained_weights, \
    read_ckpt_state_dict



class SgNetsFactory:
    @classmethod
    def get(cls, name: str, arch_params: dict = {}, checkpoint_params: dict = {}, *args,
            **kwargs) -> SgModule:
        """
        :param name: Defines the network's architecture from models/ALL_ARCHITECTURES
        :param arch_params:                Architecture hyper parameters. e.g.: block, num_blocks, num_classes, etc.
        :param checkpoint_params:          Dictionary like object with the following key:values:

            strict_load:                See StrictLoad class documentation for details.
            load_backbone:              loads the provided checkpoint to net.backbone instead of net
            checkpoint_path:   The path to the external checkpoint to be loaded. Can be absolute or relative
                                               (ie: path/to/checkpoint.pth). If provided, will automatically attempt to
                                               load the checkpoint even if the load_checkpoint flag is not provided.
            pretrained_weights: a string describing the dataset of the pretrained weights (for example "imagenent").


        """

        arch_params = core_utils.HpmStruct(**arch_params)
        checkpoint_params = core_utils.HpmStruct(**checkpoint_params)
        net = cls.instantiate_net(name, arch_params, checkpoint_params, *args, **kwargs)
        strict_load = core_utils.get_param(checkpoint_params, 'strict_load', default_val="no_key_matching")
        load_backbone = core_utils.get_param(checkpoint_params, 'load_backbone', default_val=False)
        checkpoint_path = core_utils.get_param(checkpoint_params, 'checkpoint_path')
        if checkpoint_path:
            load_ema_as_net = 'ema_net' in read_ckpt_state_dict(ckpt_path=checkpoint_path).keys()
            _ = load_checkpoint_to_model(ckpt_local_path=checkpoint_path,
                                         load_backbone=load_backbone,
                                         net=net,
                                         strict=strict_load.value if hasattr(strict_load, "value") else strict_load,
                                         load_weights_only=True,
                                         load_ema_as_net=load_ema_as_net)
        return net

    @classmethod
    def instantiate_net(cls, name: str, arch_params: dict,
                        checkpoint_params: dict) -> SgModule:
        """
        Instantiates nn.Module according to architecture and arch_params, and handles pretrained weights and the required
            module manipulation (i.e head replacement).

        :param name: Defines the network's architecture from models/ALL_ARCHITECTURES
        :param arch_params: Architecture's parameters passed to networks c'tor.
        :param checkpoint_params: checkpoint loading related parameters dictionary with 'pretrained_weights' key,
            s.t it's value is a string describing the dataset of the pretrained weights (for example "imagenent").

        :return: instantiated netowrk i.e torch.nn.Module, architecture_class (will be none when architecture is not str)

        """
        pretrained_weights = core_utils.get_param(checkpoint_params, 'pretrained_weights', default_val=None)

        if pretrained_weights is not None:
            if hasattr(arch_params, "num_classes"):
                num_classes_new_head = arch_params.num_classes
            else:
                num_classes_new_head = PRETRAINED_NUM_CLASSES[pretrained_weights]

            arch_params.num_classes = PRETRAINED_NUM_CLASSES[pretrained_weights]

        if isinstance(name, str) and name in ARCHITECTURES.keys():
            architecture_cls = ARCHITECTURES[name]
            net = architecture_cls(arch_params=arch_params)
        else:
            raise ValueError("Unsupported network name " + str(name) + ", see docs or all_architectures.py for supported "
                                                                       "nets.")
        if pretrained_weights:
            load_pretrained_weights(net, name, pretrained_weights)
            if num_classes_new_head != arch_params.num_classes:
                net.replace_head(new_num_classes=num_classes_new_head)
                arch_params.num_classes = num_classes_new_head

        return net


def get(name: str, arch_params: dict = {}, checkpoint_params: dict = {}, *args, **kwargs) -> SgModule:
    """
    :param name:               Defines the network's architecture from models/ALL_ARCHITECTURES
    :param arch_params:                Architecture hyper parameters. e.g.: block, num_blocks, num_classes, etc.
    :param checkpoint_params:          Dictionary like object with the following key:values:

            strict_load:                See StrictLoad class documentation for details.
            load_backbone:              loads the provided checkpoint to net.backbone instead of net
            checkpoint_path:   The path to the external checkpoint to be loaded. Can be absolute or relative
                                               (ie: path/to/checkpoint.pth). If provided, will automatically attempt to
                                               load the checkpoint even if the load_checkpoint flag is not provided.
            pretrained_weights: a string describing the dataset of the pretrained weights (for example "imagenent").

    """
    return SgNetsFactory.get(name, arch_params, checkpoint_params, *args, **kwargs)
