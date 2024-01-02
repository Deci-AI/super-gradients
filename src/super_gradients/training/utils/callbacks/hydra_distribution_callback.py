import torch

from super_gradients.common.registry.registry import register_callback
from super_gradients.training.utils.callbacks import PhaseContext, PhaseCallback


@register_callback("HydraDistributionVisualizationCallback")
class HydraDistributionVisualizationCallback(PhaseCallback):
    """
    HydraDistributionVisualizationCallback

    Training stage switch for Hydra training.

    """

    def __init__(self):
        super(HydraDistributionVisualizationCallback, self).__init__(phase=None)
        self.selections = []

    def on_validation_batch_end(self, context: PhaseContext) -> None:
        self.selections.append(context.preds[1])

    def on_validation_loader_end(self, context: PhaseContext) -> None:
        selections = torch.cat(self.selections, dim=0).cpu().numpy()
        if context.sg_logger is not None and not context.ddp_silent_mode:
            context.sg_logger.add_histogram(
                "distribution_histogram",
                values=selections,
                bins=5,
                global_step=context.epoch + 1,
            )

        self.selections = []
