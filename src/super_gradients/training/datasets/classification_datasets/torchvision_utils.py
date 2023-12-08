from typing import List, Any, Dict

from torchvision.datasets.vision import StandardTransform
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop, Compose

from super_gradients.common.object_names import Processings


def get_torchvision_transforms_equivalent_processing(transforms: List[Any]) -> List[Dict[str, Any]]:
    """
    Get the equivalent processing pipeline for torchvision transforms.

    :return: List of Processings operations
    """
    # Since we are using cv2.imread to read images, our model in fact is trained on BGR images.
    # In our pipelines the convention that input images are RGB, so we need to reverse the channels to get BGR
    # to match with the expected input of the model.
    pipeline = []

    if isinstance(transforms, StandardTransform):
        transforms = transforms.transform

    if isinstance(transforms, Compose):
        transforms = transforms.transforms

    for transform in transforms:
        if isinstance(transform, ToTensor):
            pipeline.append({Processings.StandardizeImage: {"max_value": 255}})
        elif isinstance(transform, Normalize):
            pipeline.append({Processings.NormalizeImage: {"mean": tuple(map(float, transform.mean)), "std": tuple(map(float, transform.std))}})
        elif isinstance(transform, Resize):
            pipeline.append({Processings.Resize: {"size": int(transform.size)}})
        elif isinstance(transform, CenterCrop):
            pipeline.append({Processings.CenterCrop: {"size": int(transform.size)}})
        else:
            raise ValueError(f"Unsupported transform: {transform}")

    pipeline.append({Processings.ImagePermute: {"permutation": (2, 0, 1)}})
    return pipeline
