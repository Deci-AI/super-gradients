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
            return self.convert_segmentation_sample_to_dict(sample, include_crowd_target="crowd_targets" in sample)

    @classmethod
    def convert_input_dict_to_segmentation_sample(cls, sample_annotations: Dict[str, Union[np.ndarray, Any]]) -> SegmentationSample:
        """
        Convert old-style detection sample dict to new DetectionSample dataclass.

        :param sample_annotations: Input dictionary with following keys:
                                    - image: numpy array of [H,W,C] or [C,H,W] format
                                    - target: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
                                    - crowd_targets: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
        :return: An instance of DetectionSample dataclass filled with data from input dictionary.
        """

        return SegmentationSample(image=sample_annotations["image"], mask=sample_annotations["mask"])

    @classmethod
    def convert_segmentation_sample_to_dict(
        cls, detection_sample: SegmentationSample, include_crowd_target: Union[bool, None]
    ) -> Dict[str, Union[np.ndarray, Any]]:
        """
        Convert new DetectionSample dataclass to old-style detection sample dict.
        This is a reverse operation to convert_input_dict_to_detection_sample and used to make legacy transforms compatible with new detection transforms.
        :param detection_sample:     Input DetectionSample dataclass.
        :param include_crowd_target: A flag indicating whether to include crowd_target in output dictionary.
                                     Can be None - in this case crowd_target will be included only if crowd targets are present in input sample.
        :return:                     Output dictionary with following keys:
                                        - image: numpy array of [H,W,C] or [C,H,W] format
                                        - target: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
                                        - crowd_targets: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
        """
        image = detection_sample.image
        crowd_mask = detection_sample.is_crowd > 0
        crowd_labels = detection_sample.labels[crowd_mask]
        crowd_bboxes_xyxy = detection_sample.bboxes_xyxy[crowd_mask]
        crowd_target = np.concatenate([crowd_bboxes_xyxy, crowd_labels[..., None]], axis=-1)

        labels = detection_sample.labels[~crowd_mask]
        bboxes_xyxy = detection_sample.bboxes_xyxy[~crowd_mask]
        target = np.concatenate([bboxes_xyxy, labels[..., None]], axis=-1)

        sample = {
            "image": image,
            "target": target,
        }
        if include_crowd_target is None:
            include_crowd_target = crowd_mask.any()
        if include_crowd_target:
            sample["crowd_target"] = crowd_target
        return sample
