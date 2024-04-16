import random

import cv2
import numpy as np

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.registry import register_transform
from super_gradients.training.samples import DetectionSample
from super_gradients.training.transforms import AbstractDetectionTransform, DetectionTransform, DetectionHSV


@register_transform("EfficientMosaicMixUpRandomAffine")
class EfficientMosaicMixUpRandomAffine(AbstractDetectionTransform):
    """
    Why MixUp "inside" Mosaic? It helps a bit. There's a thread: https://github.com/ultralytics/yolov5/issues/357
    Why efficient? It randomizes the num samples so that we get exactly how many we need, and apply transforms based on that
    Outputs can be:
    - 1 sample, with affine
    - Mosaic of 4 samples, then affine
    - MixUp of 2 Mosaics (after affine)
    """

    @resolve_param("affine_transform", TransformsFactory())
    @resolve_param("mosaic_transform", TransformsFactory())
    def __init__(self, affine_transform: DetectionTransform, mosaic_transform: DetectionTransform, mixup_prob: float):
        self.affine_transform = affine_transform
        self.mosaic_transform = mosaic_transform
        self.mosaic_prob = mosaic_transform.prob
        mosaic_transform.prob = 1.0
        self.mixup_prob = mixup_prob
        super().__init__(additional_samples_count=-1)

    @property
    def additional_samples_count(self) -> int:
        do_mosaic = random.random() < self.mosaic_prob
        do_mixup = do_mosaic and random.random() < self.mixup_prob

        return max(0, (4 * int(do_mosaic)) + (4 * int(do_mixup)) - 1)  # 0, 3, 7

    @additional_samples_count.setter
    def additional_samples_count(self, value):
        pass  # ignored

    def apply_to_sample(self, sample: DetectionSample) -> DetectionSample:
        # No MixUp, nor Mosaic
        if sample.additional_samples is None or len(sample.additional_samples) == 0:
            return self.affine_transform.apply_to_sample(sample)

        # This one for Mosaic
        maybe_samples_for_another_mosaic = sample.additional_samples[3:]
        sample.additional_samples = sample.additional_samples[:3]
        mosaic_sample = self.mosaic_transform.apply_to_sample(sample)
        mosaic_sample = self.affine_transform.apply_to_sample(mosaic_sample)

        if len(maybe_samples_for_another_mosaic) == 0:
            return mosaic_sample

        # Now we add MixUp
        new_sample, other_samples = maybe_samples_for_another_mosaic[0], maybe_samples_for_another_mosaic[1:]
        new_sample.additional_samples = other_samples
        another_mosaic = self.mosaic_transform.apply_to_sample(new_sample)
        another_mosaic = self.affine_transform.apply_to_sample(another_mosaic)

        return self._mix_two_samples(mosaic_sample, another_mosaic)

    def _mix_two_samples(self, sample_one: DetectionSample, sample_two: DetectionSample):
        mixup_boxes = np.concatenate([sample_one.bboxes_xyxy, sample_two.bboxes_xyxy], axis=0)
        mixup_labels = np.concatenate([sample_one.labels, sample_two.labels], axis=0)
        mixup_crowds = np.concatenate([sample_one.is_crowd, sample_two.is_crowd], axis=0)

        r = np.random.beta(32.0, 32.0)  # alpha=beta=32.0, see https://github.com/ultralytics/yolov5/issues/3380#issuecomment-853001307
        mixup_image = (sample_one.image * r + sample_two.image * (1 - r)).astype(sample_one.image.dtype)

        sample = DetectionSample(
            image=mixup_image,
            bboxes_xyxy=mixup_boxes,
            labels=mixup_labels,
            is_crowd=mixup_crowds,
            additional_samples=None,
        )

        return sample

    def get_equivalent_preprocessing(self):
        raise NotImplementedError("get_equivalent_preprocessing is not implemented for non-deterministic transforms.")


@register_transform("DetectionAugmentScaleHSV")
class DetectionAugmentScaleHSV(DetectionHSV):
    """
    The difference between this implementation and DetectionHSV is that this implementation *scales* the HSV while DetectionHSV *shifts* values
    Apparently, there's no best-practice there and everyone does whatever they want.
    For example: Albumentations does shift, Yolo-v8 does scale, ChatGPT suggests to shift H and scale S and V - this implementation follows v8.
    """

    def __init__(self, prob: float, hgain: float, sgain: float, vgain: float, bgr_channels=(0, 1, 2)):
        super().__init__(prob, hgain, sgain, vgain, bgr_channels)

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        return DetectionAugmentScaleHSV.augment_scale_hsv(image.copy(), self.hgain, self.sgain, self.vgain, self.bgr_channels)

    @staticmethod
    def augment_scale_hsv(img: np.ndarray, hgain: float, sgain: float, vgain: float, bgr_channels=(0, 1, 2)):
        hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1.0
        img_hsv = cv2.cvtColor(img[..., bgr_channels], cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] * hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] * hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] * hsv_augs[2], 0, 255)

        img[..., bgr_channels] = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)
        return img
