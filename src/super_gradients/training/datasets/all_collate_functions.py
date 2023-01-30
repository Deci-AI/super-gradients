from super_gradients.training.datasets.datasets_utils import ComposedCollateFunction, MultiScaleCollateFunction

ALL_COLLATE_FUNCTIONS = {
    "ComposedCollateFunction": ComposedCollateFunction,
    "MultiScaleCollateFunction": MultiScaleCollateFunction,
}
