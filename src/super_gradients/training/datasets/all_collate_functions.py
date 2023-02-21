from super_gradients.training.datasets.datasets_utils import ComposedCollateFunction, MultiScaleCollateFunction
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.training.datasets.pose_estimation_datasets import KeypointsCollate
from super_gradients.training.utils.detection_utils import DetectionCollateFN, CrowdDetectionCollateFN

ALL_COLLATE_FUNCTIONS = {
    "ComposedCollateFunction": ComposedCollateFunction,
    "MultiScaleCollateFunction": MultiScaleCollateFunction,
    "CollateMixup": CollateMixup,
    "KeypointsCollate": KeypointsCollate,
    "DetectionCollateFN": DetectionCollateFN,
    "CrowdDetectionCollateFN": CrowdDetectionCollateFN,
}
