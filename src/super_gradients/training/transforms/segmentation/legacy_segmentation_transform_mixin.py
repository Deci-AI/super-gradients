from typing import Dict, Any, Union, Tuple

import numpy as np

from super_gradients.training.samples import SegmentationSample
from PIL import Image

__all__ = ["LegacySegmentationTransformMixin"]


class LegacySegmentationTransformMixin:
    """
    A mixin class to make legacy detection transforms compatible with new detection transforms that operate on DetectionSample.
    """

    def __call__(self, sample: Union["SegmentationSample", Dict[str, Any]]) -> Union["SegmentationSample", Dict[str, Any]]:
        """
        :param sample: Dict with following keys:
                        - image: numpy array of [H,W,C] or [C,H,W] format
                        - target: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
                        - crowd_targets: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
        """

        if isinstance(sample, SegmentationSample):
            return self.apply_to_sample(sample)
        else:
            sample, image_is_pil = self.convert_input_dict_to_segmentation_sample(sample)
            sample = self.apply_to_sample(sample)
            return self.convert_segmentation_sample_to_dict(sample, image_is_pil)

    @classmethod
    def convert_input_dict_to_segmentation_sample(cls, sample_annotations: Dict[str, Union[np.ndarray, Any]]) -> Tuple[SegmentationSample, bool]:
        """
        Convert old-style segmentation sample dict to new DetectionSample dataclass.

        :param sample_annotations: Input dictionary with following keys:
            image:              Associated image with sample, in [H,W,C] (or H,W for greyscale) format.
            mask:               Associated segmentation mask with sample, in [H,W]

        :return: A tuple of SegmentationSample and a boolean value indicating whether original input dict has images as PIL Image
                 An instance of SegmentationSample dataclass filled with data from input dictionary.
        """

        image_is_pil = isinstance(sample_annotations["image"], Image.Image)
        return SegmentationSample(image=sample_annotations["image"], mask=sample_annotations["mask"]), image_is_pil

    @classmethod
    def convert_segmentation_sample_to_dict(cls, sample: SegmentationSample, image_is_pil: bool) -> Dict[str, Union[np.ndarray, Any]]:
        """
        Convert new SegmentationSample dataclass to old-style detection sample dict. This is a reverse operation to
        convert_input_dict_to_detection_sample and used to make legacy transforms compatible with new segmentation
        transforms.
        :param sample:       Transformed sample
        :param image_is_pil: A boolean value indicating whether original input dict has images as PIL Image
                             If True, output dict will also have images as PIL Image, otherwise as numpy array.
        """

        image = sample.image
        mask = sample.mask

        if image_is_pil:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
        return {"image": image, "mask": mask}
