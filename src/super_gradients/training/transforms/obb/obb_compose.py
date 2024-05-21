from typing import List

from .abstract_obb_transform import AbstractOBBDetectionTransform
from super_gradients.training.samples.obb_sample import OBBSample


class OBBDetectionCompose(AbstractOBBDetectionTransform):
    """
    Composes several transforms together
    """

    def __init__(self, transforms: List[AbstractOBBDetectionTransform], load_sample_fn=None):
        """

        :param transforms:         List of keypoint-based transformations
        :param load_sample_fn:     A method to load additional samples if needed (for mixup & mosaic augmentations).
                                   Default value is None, which would raise an error if additional samples are needed.
        """
        for transform in transforms:
            if hasattr(transform, "may_require_additional_samples") and transform.may_require_additional_samples and load_sample_fn is None:
                raise RuntimeError(f"Transform {transform.__class__.__name__} that requires additional samples but `load_sample_fn` is None")

        super().__init__()
        self.transforms = transforms
        self.load_sample_fn = load_sample_fn

    def apply_to_sample(self, sample: OBBSample) -> OBBSample:
        """
        Applies the series of transformations to the input sample.
        The function may modify the input sample inplace, so input sample should not be used after the call.

        :param sample: Input sample
        :return:       Transformed sample.
        """
        sample = sample.sanitize_sample()
        sample = self._apply_transforms(sample, transforms=self.transforms, load_sample_fn=self.load_sample_fn)
        return sample

    @classmethod
    def _apply_transforms(cls, sample: OBBSample, transforms: List[AbstractOBBDetectionTransform], load_sample_fn) -> OBBSample:
        """
        This helper method allows us to query additional samples for mixup & mosaic augmentations
        that would be also passed through augmentation pipeline. Example:

        ```
          transforms:
            - OBBDetectionLongestMaxSize:
                max_height: ${dataset_params.image_size}
                max_width: ${dataset_params.image_size}
            - OBBDetectionMixup:
                prob: ${dataset_params.mixup_prob}
        ```

        In the example above all samples in mixup will be forwarded through OBBDetectionLongestMaxSize,
        and only then mixed up.

        :param sample:         Input data sample
        :param transforms:     List of transformations to apply
        :param load_sample_fn: A method to load additional samples if needed
        :return:               A data sample after applying transformations
        """
        applied_transforms_so_far = []
        for t in transforms:
            if not hasattr(t, "may_require_additional_samples") or not t.may_require_additional_samples:
                sample = t.apply_to_sample(sample)
                applied_transforms_so_far.append(t)
            else:
                additional_samples = [load_sample_fn() for _ in range(t.get_number_of_additional_samples())]
                additional_samples = [
                    cls._apply_transforms(
                        sample,
                        applied_transforms_so_far,
                        load_sample_fn=load_sample_fn,
                    )
                    for sample in additional_samples
                ]
                sample.additional_samples = additional_samples
                sample = t.apply_to_sample(sample)

        return sample

    def get_equivalent_preprocessing(self) -> List:
        preprocessing = []
        for t in self.transforms:
            preprocessing += t.get_equivalent_preprocessing()
        return preprocessing

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\t{repr(t)}"
        format_string += "\n)"
        return format_string
