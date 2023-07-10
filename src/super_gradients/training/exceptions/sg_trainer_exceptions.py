from typing import List


class UnsupportedTrainingParameterFormat(Exception):
    """Exception raised illegal training param format.

    :param desc: Explanation of the error
    """

    def __init__(self, desc: str):
        self.message = "Unsupported training parameter format: " + desc
        super().__init__(self.message)


class UnsupportedOptimizerFormat(UnsupportedTrainingParameterFormat):
    """Exception raised illegal optimizer format."""

    def __init__(self):
        super().__init__("optimizer parameter expected one of ['Adam','SGD','RMSProp'], or torch.optim.Optimizer object")


class IllegalDataloaderInitialization(Exception):
    """Exception raised illegal data loaders."""

    def __init__(self):
        super().__init__("train_loader, valid_loader and class parameters are required when initializing Trainer with data loaders")


class GPUModeNotSetupError(Exception):
    """Exception raised when the DDP should be setup but is not."""

    def __init__(self):
        super().__init__(
            "Your environment was not setup to support DDP. Please run at the beginning of your script:\n"
            ">>> from super_gradients.common.environment.env_helpers import init_trainer\n"
            ">>> setup_device(multi_gpu=..., num_gpus=...)\n"
        )


class IllegalMetricToWatch(Exception):
    def __init__(self, metric_to_watch: str, loss_component_names: List[str], metric_titles: List[str]):
        self.loss_component_names = loss_component_names
        self.metric_titles = metric_titles
        self.metric_to_watch = metric_to_watch
        super(IllegalMetricToWatch, self).__init__(
            f"metric_to_watch: {self.metric_to_watch} not in possible monitored values: {self.loss_component_names + self.metric_titles}"
        )
