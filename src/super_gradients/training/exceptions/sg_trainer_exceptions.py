
class UnsupportedTrainingParameterFormat(Exception):
    """Exception raised illegal training param format.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, desc):
        self.message = "Unsupported training parameter format: " + desc
        super().__init__(self.message)


class UnsupportedOptimizerFormat(UnsupportedTrainingParameterFormat):
    """Exception raised illegal optimizer format.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        super().__init__(
            "optimizer parameter expected one of ['Adam','SGD','RMSProp'], or torch.optim.Optimizer object")


class IllegalDataloaderInitialization(Exception):
    """Exception raised illegal data loaders.
    """

    def __init__(self):
        super().__init__(
            "train_loader, valid_loader and class parameters are required when initializing Trainer with data loaders")
