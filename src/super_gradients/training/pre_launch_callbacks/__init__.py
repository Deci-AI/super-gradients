from super_gradients.training.pre_launch_callbacks.pre_launch_callbacks import (
    PreLaunchCallback,
    AutoTrainBatchSizeSelectionCallback,
    QATRecipeModificationCallback,
)

ALL_PRE_LAUNCH_CALLBACKS = {
    "AutoTrainBatchSizeSelectionCallback": AutoTrainBatchSizeSelectionCallback,
    "QATRecipeModificationCallback": QATRecipeModificationCallback,
}

__all__ = ["PreLaunchCallback", "AutoTrainBatchSizeSelectionCallback", "QATRecipeModificationCallback", "ALL_PRE_LAUNCH_CALLBACKS"]
