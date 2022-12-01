from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.training.utils.optimizers import OPTIMIZERS


class OptimizersTypeFactory(TypeFactory):
    """
    This is a special factory for torch.optim.Optimizer.
    This factory does not instantiate an object but rather return the type, since optimizer instantiation
    requires the model to be instantiated first
    """

    def __init__(self):
        super().__init__(OPTIMIZERS)
