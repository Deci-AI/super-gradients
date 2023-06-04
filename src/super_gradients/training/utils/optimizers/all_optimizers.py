import warnings

from super_gradients.common.registry.registry import OPTIMIZERS as CURRENT_VERSION_OPTIMIZERS

OPTIMIZERS = CURRENT_VERSION_OPTIMIZERS

warnings.warn(
    "super_gradients.training.utils.optimizers.all_opimitzers is deprecated in 3.1.1 and will be removed in 3.2.0.\n"
    " To import OPTIMIZERS use: \n from super_gradients.common.registry.registry import OPTIMIZERS",
    category=DeprecationWarning,
)
