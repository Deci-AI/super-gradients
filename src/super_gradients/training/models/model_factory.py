from super_gradients.common import StrictLoad
from super_gradients.training import utils as core_utils
from super_gradients.training.models import SgModule
from super_gradients.training.models.all_architectures import ARCHITECTURES
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model, load_pretrained_weights, \
    read_ckpt_state_dict
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def instantiate_model(name: str, arch_params: dict, pretrained_weights: str = None) -> SgModule:
    """
    Instantiates nn.Module according to architecture and arch_params, and handles pretrained weights and the required
        module manipulation (i.e head replacement).

    :param name: Defines the model's architecture from models/ALL_ARCHITECTURES
    :param arch_params: Architecture's parameters passed to models c'tor.
    :param pretrained_weights: string describing the dataset of the pretrained weights (for example "imagenent")

    :return: instantiated model i.e torch.nn.Module, architecture_class (will be none when architecture is not str)

    """

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
            "Unsupported model name " + str(name) + ", see docs or all_architectures.py for supported "
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
    :param name:               Defines the model's architecture from models/ALL_ARCHITECTURES
    :param num_classes:        Number of classes (defines the net's structure). If None is given, will try to derrive from
                                pretrained_weight's corresponding dataset.
    :param arch_params:                Architecture hyper parameters. e.g.: block, num_blocks, etc.

    :param strict_load:                See super_gradients.common.data_types.enum.strict_load.StrictLoad class documentation for details
     (default=NO_KEY_MATCHING to suport SG trained checkpoints)
    :param load_backbone:              loads the provided checkpoint to model.backbone instead of model.
    :param checkpoint_path:   The path to the external checkpoint to be loaded. Can be absolute or relative
                                       (ie: path/to/checkpoint.pth). If provided, will automatically attempt to
                                       load the checkpoint.
    :param pretrained_weights: a string describing the dataset of the pretrained weights (for example "imagenent").

    NOTE: Passing pretrained_weights and checkpoint_path is ill-defined and will raise an error.

    """
    if arch_params.get("num_classes") is not None:
        logger.warning("Passing num_classes through arch_params is dperecated and will be removed in the next version. "
                       "Pass num_classes explicitly to models.get")
    num_classes = num_classes or arch_params.get("num_classes")

    if pretrained_weights is None and num_classes is None:
        raise ValueError("num_classes or pretrained_weights must be passed to determine net's structure.")

    if num_classes is not None:
        arch_params["num_classes"] = num_classes

    arch_params = core_utils.HpmStruct(**arch_params)
    net = instantiate_model(name, arch_params, pretrained_weights)

    if checkpoint_path:
        load_ema_as_net = 'ema_net' in read_ckpt_state_dict(ckpt_path=checkpoint_path).keys()
        _ = load_checkpoint_to_model(ckpt_local_path=checkpoint_path,
                                     load_backbone=load_backbone,
                                     net=net,
                                     strict=strict_load.value if hasattr(strict_load, "value") else strict_load,
                                     load_weights_only=True,
                                     load_ema_as_net=load_ema_as_net)
    return net
