from torch import optim
from super_gradients.common.object_names import Optimizers
from super_gradients.training.utils.optimizers.rmsprop_tf import RMSpropTF
from super_gradients.training.utils.optimizers.lamb import Lamb


OPTIMIZERS = {
    Optimizers.SGD: optim.SGD,
    Optimizers.ADAM: optim.Adam,
    Optimizers.RMS_PROP: optim.RMSprop,
    Optimizers.RMS_PROP_TF: RMSpropTF,
    Optimizers.LAMB: Lamb
}
