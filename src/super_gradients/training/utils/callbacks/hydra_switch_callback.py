from super_gradients.common.registry.registry import register_callback
from super_gradients.training.utils.callbacks import TrainingStageSwitchCallbackBase, PhaseContext


@register_callback("HydraTrainingStageSwitchCallback")
class HydraTrainingStageSwitchCallback(TrainingStageSwitchCallbackBase):
    """
    HydraTrainingStageSwitchCallback

    Training stage switch for Hydra training.

    """

    def __init__(
        self,
        next_stage_start_epoch: int = 15,
    ):
        super().__init__(next_stage_start_epoch=next_stage_start_epoch)

    def apply_stage_change(self, context: PhaseContext):
        if hasattr(context.net, "module"):
            context.net.module.set_phase(2)
        else:
            context.net.set_phase(2)

        context.criterion.set_phase(2)
