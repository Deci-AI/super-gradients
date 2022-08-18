from deprecate import deprecated

from super_gradients.training import Trainer


@deprecated(target=Trainer, deprecated_in='2.3.0', remove_in='3.0.0')
class SgModel(Trainer):
    def __init__(self, experiment_name: str, *args, **kwargs):
        super().__init__(experiment_name, *args, **kwargs)
