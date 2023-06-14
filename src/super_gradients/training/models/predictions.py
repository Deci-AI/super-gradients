from super_gradients.training.utils.predict import Prediction, DetectionPrediction
import warnings

warnings.warn(
    "Importing from super_gradients.training.models.predictions is deprecated. "
    "Please update your code to import from super_gradients.training.utils.predict instead.",
    DeprecationWarning,
)


__all__ = ["Prediction", "DetectionPrediction"]
