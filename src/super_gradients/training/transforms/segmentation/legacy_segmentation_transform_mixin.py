from typing import Dict, Any, Union

import numpy as np

from super_gradients.training.samples import SegmentationSample


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
            sample = self.convert_input_dict_to_segmentation_sample(sample)
            sample = self.apply_to_sample(sample)
            return self.convert_segmentation_sample_to_dict(sample)

    @classmethod
    def convert_input_dict_to_segmentation_sample(cls, sample_annotations: Dict[str, Union[np.ndarray, Any]]) -> SegmentationSample:
        """
        Convert old-style segmentation sample dict to new DetectionSample dataclass.

        :param sample_annotations: Input dictionary with following keys:
            image:              Associated image with sample, in [H,W,C] (or H,W for greyscale) format.
            mask:               Associated segmentation mask with sample, in [H,W]

        :return: An instance of SegmentationSample dataclass filled with data from input dictionary.
        """

        return SegmentationSample(image=sample_annotations["image"], mask=sample_annotations["mask"])

    @classmethod
    def convert_segmentation_sample_to_dict(cls, sample: SegmentationSample) -> Dict[str, Union[np.ndarray, Any]]:
        """
        Convert new SegmentationSample dataclass to old-style detection sample dict. This is a reverse operation to
        convert_input_dict_to_detection_sample and used to make legacy transforms compatible with new segmentation
        transforms. :param sample:
        """

        return {"image": sample.image, "mask": sample.mask}
