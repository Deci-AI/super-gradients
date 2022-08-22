from super_gradients.common import StrictLoad
from super_gradients.training import utils as core_utils
from super_gradients.training.models import SgModule
from super_gradients.training.models.all_architectures import ARCHITECTURES
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model, load_pretrained_weights, \
    read_ckpt_state_dict
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


class SgNetsFactory:
    @classmethod
    def get(cls, name: str, arch_params: dict = {}, checkpoint_params: dict = {}) -> SgModule:
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
        net = cls.instantiate_net(name, arch_params, checkpoint_params)
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
            raise ValueError(
                "Unsupported network name " + str(name) + ", see docs or all_architectures.py for supported "
                                                          "nets.")
        if pretrained_weights:
            load_pretrained_weights(net, name, pretrained_weights)
            if num_classes_new_head != arch_params.num_classes:
                net.replace_head(new_num_classes=num_classes_new_head)
                arch_params.num_classes = num_classes_new_head

        return net


def get(name: str, arch_params: dict = {}, num_classes: int = None,
        strict_load: StrictLoad = StrictLoad.NO_KEY_MATCHING, checkpoint_path: str = None,
        pretrained_weights: str = None, load_backbone: bool = False) -> SgModule:
    """
    :param name:               Defines the network's architecture from models/ALL_ARCHITECTURES
    :param num_classes:        Number of classes (defines the net's structure). If None is given, will try to derrive from
                                pretrained_weight's corresponding dataset.
    :param arch_params:                Architecture hyper parameters. e.g.: block, num_blocks, etc.

    :param strict_load:                See StrictLoad class documentation for details (default=NO_KEY_MATCHING to suport SG trained checkpoints)
    :param load_backbone:              loads the provided checkpoint to net.backbone instead of net
    :param checkpoint_path:   The path to the external checkpoint to be loaded. Can be absolute or relative
                                       (ie: path/to/checkpoint.pth). If provided, will automatically attempt to
                                       load the checkpoint.
    :param pretrained_weights: a string describing the dataset of the pretrained weights (for example "imagenent").

    """
    if arch_params.get("num_classes") is not None:
        logger.warning("Passing num_classes through arch_params is dperecated and will be removed in the next version. "
                       "Pass num_classes explicitly to models.get")
    num_classes = num_classes or arch_params.get("num_classes")
    if pretrained_weights is None and num_classes is None:
        raise ValueError("num_classes or pretrained_weights must be passed to determine net's structure.")

    checkpoint_params = {"strict_load": strict_load,
                         "checkpoint_path": checkpoint_path,
                         "load_backbone": load_backbone,
                         "pretrained_weights": pretrained_weights}

    if num_classes is not None:
        arch_params["num_classes"] = num_classes

    return SgNetsFactory.get(name, arch_params, checkpoint_params)
