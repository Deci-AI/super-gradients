import numpy as np

from super_gradients.training.datasets.datasets_conf import PASCAL_VOC_2012_CLASSES_LIST


def pascalvoc_target_adapter(img_annotations: dict) -> np.ndarray:
    """Parse torchvision.datasets.VOCDetection image annotations to only return its targets in format expected by DetectionTransforms.

    :param img_annotations: Refers to VOCDetection(...).__getitem__()[1]
    :return:                Target in XYXY_LABEL, which is expected by DetectionTransforms
    """

    annotations = img_annotations["annotation"]["object"]
    target = np.zeros((len(annotations), 5))
    for ix, annotation in enumerate(annotations):
        bbox = annotation["bndbox"]
        target[ix, 0:4] = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])
        cls_id = PASCAL_VOC_2012_CLASSES_LIST.index(annotation["name"])
        target[ix, 4] = cls_id
    return target
