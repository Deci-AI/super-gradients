__all__ = ["LegacyDetectionTransformMixin"]

from typing import Dict, Any

import numpy as np

from super_gradients.training.samples import DetectionSample


class LegacyDetectionTransformMixin:
    """
    A mixin class to make legacy detection transforms compatible with new detection transforms that operate on DetectionSample.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param sample: Dict with following keys:
                        - image: numpy array of [H,W,C] or [C,H,W] format
                        - target: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
                        - crowd_targets: numpy array of [N,5] shape with bounding box of each instance (XYXY + LABEL)
        """

        image, targets = sample["image"], sample["target"]
        if len(targets) == 0:
            targets = np.zeros((0, 5), dtype=np.float32)
        is_crowd = np.zeros(len(targets), dtype=bool)

        crowd_targets = sample.get("crowd_targets")
        if crowd_targets is not None:
            if len(crowd_targets) == 0:
                crowd_targets = np.zeros((0, 5), dtype=np.float32)

            targets = np.concatenate([targets, crowd_targets], axis=0)
            is_crowd = np.concatenate([is_crowd, np.ones(len(crowd_targets), dtype=bool)], axis=0)

        sample = DetectionSample(
            image=image,
            bboxes_xyxy=targets[:, :4],
            labels=targets[:, 4:],
            is_crowd=is_crowd,
            additional_samples=None,
        )

        sample = self.apply_to_sample(sample)

        all_targets = np.concatenate([sample.bboxes_xyxy, sample.labels], axis=1)
        is_crowd = sample.is_crowd

        return {
            "image": sample.image,
            "target": all_targets[~is_crowd],
            "crowd_targets": all_targets[is_crowd],
        }
