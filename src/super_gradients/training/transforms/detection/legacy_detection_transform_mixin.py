from typing import Dict, Any, Union

import numpy as np

from super_gradients.training.samples import DetectionSample


__all__ = ["LegacyDetectionTransformMixin"]


class LegacyDetectionTransformMixin:
    """
    A mixin class to make legacy detection transforms compatible with new detection transforms that operate on DetectionSample.
    """

    def __call__(self, sample: Union["DetectionSample", Dict[str, Any]]) -> Union["DetectionSample", Dict[str, Any]]:
        """
        :param sample: Dict with following keys:
                        - image: numpy array of [H,W,C] or [C,H,W] format
                        - target: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
                        - crowd_targets: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
        """

        if isinstance(sample, DetectionSample):
            return self.apply_to_sample(sample)
        else:
            has_crowd_target = "crowd_target" in sample
            sample = self.convert_input_dict_to_detection_sample(sample)
            sample = self.apply_to_sample(sample)
            return self.convert_detection_sample_to_dict(sample, include_crowd_target=has_crowd_target)

    @classmethod
    def convert_input_dict_to_detection_sample(cls, sample_annotations: Dict[str, Union[np.ndarray, Any]]) -> DetectionSample:
        """
        Convert old-style detection sample dict to new DetectionSample dataclass.

        :param sample_annotations: Input dictionary with following keys:
                                    - image: numpy array of [H,W,C] or [C,H,W] format
                                    - target: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
                                    - crowd_targets: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
        :return: An instance of DetectionSample dataclass filled with data from input dictionary.
        """
        target = sample_annotations["target"]
        if len(target) == 0:
            target = np.zeros((0, 5), dtype=np.float32)

        bboxes_xyxy = target[:, 0:4].reshape(-1, 4)
        labels = target[:, 4]

        is_crowd = np.zeros_like(labels, dtype=bool)
        if "crowd_target" in sample_annotations:
            crowd_target = sample_annotations["crowd_target"]
            if len(crowd_target) == 0:
                crowd_target = np.zeros((0, 5), dtype=np.float32)

            crowd_bboxes_xyxy = crowd_target[:, 0:4].reshape(-1, 4)
            crowd_labels = crowd_target[:, 4]
            bboxes_xyxy = np.concatenate([bboxes_xyxy, crowd_bboxes_xyxy], axis=0)
            labels = np.concatenate([labels, crowd_labels], axis=0)
            is_crowd = np.concatenate([is_crowd, np.ones_like(crowd_labels, dtype=bool)], axis=0)

        return DetectionSample(
            image=sample_annotations["image"],
            bboxes_xyxy=bboxes_xyxy,
            labels=labels,
            is_crowd=is_crowd,
            additional_samples=None,
        )

    @classmethod
    def convert_detection_sample_to_dict(cls, detection_sample: DetectionSample, include_crowd_target: Union[bool, None]) -> Dict[str, Union[np.ndarray, Any]]:
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
