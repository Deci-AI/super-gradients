from super_gradients.common.object_names import CollateFunctions
from torch.utils.data._utils.collate import default_collate
from super_gradients.training.utils.collate_fn.detection import DetectionCollateFN, CrowdDetectionCollateFN


COLLATE_FUNCTIONS = {
    CollateFunctions.DEFAULT_COLLATE_FN: default_collate,
    CollateFunctions.DETECTION_COLLATE_FN: DetectionCollateFN,
    CollateFunctions.CROWD_DETECTION_COLLATE_FN: CrowdDetectionCollateFN,
}
