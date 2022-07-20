from deprecate import deprecated

from super_gradients import Trainer

@deprecated(target=Trainer, deprecated_in='2.2.0', remove_in='2.6.0')
class SG_Model(Trainer):
    def __init__(self, experiment_name: str, *args, **kwargs):
        super().__init__(experiment_name,  *args, **kwargs)