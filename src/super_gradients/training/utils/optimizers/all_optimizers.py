from torch import optim
from super_gradients.training.utils.optimizers.rmsprop_tf import RMSpropTF
from super_gradients.training.utils.optimizers.lamb import Lamb


class OptimizerNames:
    SGD = "SGD"
    ADAM = "Adam"
    RMS_PROP = "RMSprop"
    RMS_PROP_TF = "RMSpropTF"
    LAMB = "Lamb"


OPTIMIZERS = {
    OptimizerNames.SGD: optim.SGD,
    OptimizerNames.ADAM: optim.Adam,
    OptimizerNames.RMS_PROP: optim.RMSprop,
    OptimizerNames.RMS_PROP_TF: RMSpropTF,
    OptimizerNames.LAMB: Lamb
}
