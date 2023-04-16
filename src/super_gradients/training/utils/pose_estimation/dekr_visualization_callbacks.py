import cv2
from typing import List

import numpy as np
import torch
from torch import Tensor

from super_gradients.common.registry.registry import register_callback
from super_gradients.common.object_names import Callbacks
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from super_gradients.training.utils.pose_estimation.dekr_decode_callbacks import _hierarchical_pool
from super_gradients.common.environment.ddp_utils import multi_process_safe

__all__ = ["DEKRVisualizationCallback"]


@register_callback(Callbacks.DEKR_VISUALIZATION)
class DEKRVisualizationCallback(PhaseCallback):
    """
    A callback that adds a visualization of a batch of segmentation predictions to context.sg_logger

    :param phase:                   When to trigger the callback.
    :param prefix:                  Prefix to add to the log.
    :param mean:                    Mean to subtract from image.
    :param std:                     Standard deviation to subtract from image.
    :param apply_sigmoid:           Whether to apply sigmoid to the output.
    :param batch_idx:               Batch index to perform visualization for.
    :param keypoints_threshold:     Keypoint threshold to use for visualization.
    """

    def __init__(
        self,
        phase: Phase,
        prefix: str,
        mean: List[float],
        std: List[float],
        apply_sigmoid: bool = False,
        batch_idx: int = 0,
        keypoints_threshold: float = 0.01,
    ):
        super(DEKRVisualizationCallback, self).__init__(phase)
        self.batch_idx = batch_idx
        self.prefix = prefix
        self.mean = np.array(list(map(float, mean))).reshape((1, 1, -1))
        self.std = np.array(list(map(float, std))).reshape((1, 1, -1))
        self.apply_sigmoid = apply_sigmoid
        self.keypoints_threshold = keypoints_threshold

    def denormalize_image(self, image_normalized: Tensor) -> np.ndarray:
        """
        Reverse image normalization image_normalized (image / 255 - mean) / std
        :param image_normalized: normalized [3,H,W]
        :return:
        """

        image_normalized = torch.moveaxis(image_normalized, 0, -1).detach().cpu().numpy()
        image = (image_normalized * self.std + self.mean) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)[..., ::-1]
        return image

    @classmethod
    def visualize_heatmap(self, heatmap: Tensor, apply_sigmoid: bool, dsize, min_value=None, max_value=None, colormap=cv2.COLORMAP_JET):
        if apply_sigmoid:
            heatmap = heatmap.sigmoid()

        if min_value is None:
            min_value = heatmap.min().item()

        if max_value is None:
            max_value = heatmap.max().item()

        heatmap = heatmap.detach().cpu().numpy()
        real_min = heatmap.min()
        real_max = heatmap.max()

        heatmap = np.max(heatmap, axis=0)
        heatmap = (heatmap - min_value) / (1e-8 + max_value - min_value)
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_8u = (heatmap * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_8u, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        if dsize is not None:
            heatmap_rgb = cv2.resize(heatmap_rgb, dsize=dsize)

        cv2.putText(
            heatmap_rgb,
            f"min:{real_min:.3f}",
            (5, 15),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            color=(255, 255, 255),
            fontScale=0.8,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            heatmap_rgb,
            f"max:{real_max:.3f}",
            (5, heatmap_rgb.shape[0] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            color=(255, 255, 255),
            fontScale=0.8,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        return heatmap, heatmap_rgb

    @multi_process_safe
    def __call__(self, context: PhaseContext):
        if context.batch_idx == self.batch_idx:
            batch_imgs = self.visualize_batch(context.inputs, context.preds, context.target)
            batch_imgs = np.stack(batch_imgs)
            tag = self.prefix + str(self.batch_idx) + "_images"
            context.sg_logger.add_images(tag=tag, images=batch_imgs, global_step=context.epoch, data_format="NHWC")

    @torch.no_grad()
    def visualize_batch(self, inputs, predictions, targets):
        num_samples = len(inputs)
        batch_imgs = []

        gt_heatmap, mask, _, _ = targets

        # Check whether model also produce supervised output predictions
        if isinstance(predictions, tuple) and len(predictions) == 2 and torch.is_tensor(predictions[0]) and torch.is_tensor(predictions[1]):
            heatmap, _ = predictions
        else:
            (heatmap, _), (_, _) = predictions

        for i in range(num_samples):
            batch_imgs.append(self.visualize_sample(inputs[i], predicted_heatmap=heatmap[i], target_heatmap=gt_heatmap[i], target_mask=mask[i]))

        return batch_imgs

    def visualize_sample(self, input, predicted_heatmap, target_heatmap, target_mask):
        image_rgb = self.denormalize_image(input)
        dsize = image_rgb.shape[1], image_rgb.shape[0]
        half_size = dsize[0] // 2, dsize[1] // 2

        target_heatmap_f32, target_heatmap_rgb = self.visualize_heatmap(target_heatmap, apply_sigmoid=False, dsize=half_size)
        target_heatmap_f32 = cv2.resize(target_heatmap_f32, dsize=dsize)
        target_heatmap_f32 = np.expand_dims(target_heatmap_f32, -1)

        peaks_heatmap = _hierarchical_pool(predicted_heatmap)[0]
        peaks_heatmap = predicted_heatmap.eq(peaks_heatmap) & (predicted_heatmap > self.keypoints_threshold)

        peaks_heatmap = peaks_heatmap.sum(dim=0, keepdim=False) > 0

        # Apply masking with GT mask to suppress predictions on ignored areas of the image (where target_mask==0)
        flat_target_mask = target_mask.sum(dim=0, keepdim=False) > 0
        peaks_heatmap &= flat_target_mask
        peaks_heatmap = peaks_heatmap.detach().cpu().numpy().astype(np.uint8) * 255

        peaks_heatmap = cv2.applyColorMap(peaks_heatmap, cv2.COLORMAP_JET)
        peaks_heatmap = cv2.cvtColor(peaks_heatmap, cv2.COLOR_BGR2RGB)
        peaks_heatmap = cv2.resize(peaks_heatmap, dsize=half_size)

        _, predicted_heatmap_rgb = self.visualize_heatmap(
            predicted_heatmap, min_value=target_heatmap.min().item(), max_value=target_heatmap.max().item(), apply_sigmoid=self.apply_sigmoid, dsize=half_size
        )

        image_heatmap_overlay = image_rgb * (1 - target_heatmap_f32) + target_heatmap_f32 * cv2.resize(target_heatmap_rgb, dsize=dsize)
        image_heatmap_overlay = image_heatmap_overlay.astype(np.uint8)

        _, target_mask_rgb = self.visualize_heatmap(target_mask, min_value=0, max_value=1, apply_sigmoid=False, dsize=half_size, colormap=cv2.COLORMAP_BONE)

        return np.hstack(
            [
                image_heatmap_overlay,
                np.vstack([target_heatmap_rgb, predicted_heatmap_rgb]),
                np.vstack([target_mask_rgb, peaks_heatmap]),
            ]
        )
