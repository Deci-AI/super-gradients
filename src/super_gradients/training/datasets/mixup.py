""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
CutMix by timm: https://github.com/rwightman/pytorch-image-models/timm

"""
from typing import List, Union

import numpy as np
import torch

from super_gradients.common.registry.registry import register_collate_function
from super_gradients.training.exceptions.dataset_exceptions import IllegalDatasetParameterException


def one_hot(x, num_classes, on_value=1.0, off_value=0.0, device="cuda"):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target: torch.Tensor, num_classes: int, lam: float = 1.0, smoothing: float = 0.0, device: str = "cuda"):
    """
    generate a smooth target (label) two-hot tensor to support the mixed images with different labels
    :param target: the targets tensor
    :param num_classes: number of classes (to set the final tensor size)
    :param lam: percentage of label a range [0, 1] in the mixing
    :param smoothing: the smoothing multiplier
    :param device: usable device ['cuda', 'cpu']
    :return:
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1.0 - lam)


def rand_bbox(img_shape: tuple, lam: float, margin: float = 0.0, count: int = None):
    """Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    :param img_shape: Image shape as tuple
    :param lam: Cutmix lambda value
    :param margin: Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
    :param count: Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape: tuple, minmax: Union[tuple, list], count: int = None):
    """Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    :param img_shape: Image shape as tuple
    :param minmax: Min and max bbox ratios (as percent of image size)
    :param count: Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape: tuple, lam: float, ratio_minmax: Union[tuple, list] = None, correct_lam: bool = True, count: int = None):
    """
    Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


@register_collate_function()
class CollateMixup:
    """
    Collate with Mixup/Cutmix that applies different params to each element or whole batch
    A Mixup impl that's performed while collating the batches.
    """

    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 0.0,
        cutmix_minmax: List[float] = None,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = "batch",
        correct_lam: bool = True,
        label_smoothing: float = 0.1,
        num_classes: int = 1000,
    ):
        """
        Mixup/Cutmix that applies different params to each element or whole batch

        :param mixup_alpha: mixup alpha value, mixup is active if > 0.
        :param cutmix_alpha: cutmix alpha value, cutmix is active if > 0.
        :param cutmix_minmax: cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        :param prob: probability of applying mixup or cutmix per batch or element
        :param switch_prob: probability of switching to cutmix instead of mixup when both are active
        :param mode: how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        :param correct_lam: apply lambda correction when cutmix bbox clipped by image borders
        :param label_smoothing: apply label smoothing to the mixed target tensor
        :param num_classes: number of classes for target
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        """
        generate two random masks to define which elements of the batch will be mixed and how (depending on the
        self.mixup_enabled, self.mixup_alpha, self.cutmix_alpha parameters

        :param batch_size:
        :return: two tensors with shape=batch_size - the first contains the lambda value per batch element
        and the second is a binary flag indicating use of cutmix per batch element
        """
        lam = torch.ones(batch_size, dtype=torch.float32)
        use_cutmix = torch.zeros(batch_size, dtype=torch.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = torch.rand(batch_size) < self.switch_prob
                lam_mix = torch.where(
                    use_cutmix,
                    torch.distributions.beta.Beta(self.cutmix_alpha, self.cutmix_alpha).sample(sample_shape=batch_size),
                    torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample(sample_shape=batch_size),
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample(sample_shape=batch_size)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = torch.ones(batch_size, dtype=torch.bool)
                lam_mix = torch.distributions.beta.Beta(self.cutmix_alpha, self.cutmix_alpha).sample(sample_shape=batch_size)
            else:
                raise IllegalDatasetParameterException("One of mixup_alpha > 0., cutmix_alpha > 0., " "cutmix_minmax not None should be true.")
            lam = torch.where(torch.rand(batch_size) < self.mix_prob, lam_mix.type(torch.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        """
        generate two random parameters to define if batch will be mixed and how (depending on the
        self.mixup_enabled, self.mixup_alpha, self.cutmix_alpha parameters

        :return: two parameters - the first contains the lambda value for the whole batch
        and the second is a binary flag indicating use of cutmix for the batch
        """
        lam = 1.0
        use_cutmix = False

        if self.mixup_enabled and torch.rand(1) < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = torch.rand(1) < self.switch_prob
                lam_mix = (
                    torch.distributions.beta.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
                    if use_cutmix
                    else torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample()
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample()
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = torch.distributions.beta.Beta(self.cutmix_alpha, self.cutmix_alpha).sample()
            else:
                raise IllegalDatasetParameterException("One of mixup_alpha > 0., cutmix_alpha > 0., " "cutmix_minmax not None should be true.")
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem_collate(self, output: torch.Tensor, batch: list, half: bool = False):
        """
        This is the implementation for 'elem' or 'half' modes
        :param output: the output tensor to fill
        :param batch: list of thr batch items
        :return: a tensor containing the lambda values used for the mixing (this vector can be used for
        mixing the labels as well)
        """
        batch_size = len(batch)
        num_elem = batch_size // 2 if half else batch_size
        assert len(output) == num_elem
        lam_batch, use_cutmix = self._params_per_elem(num_elem)
        for i in range(num_elem):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed = batch[i][0]
            if lam != 1.0:
                if use_cutmix[i]:
                    if not half:
                        mixed = torch.clone(mixed)
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    mixed = mixed * lam + batch[j][0] * (1 - lam)
            output[i] += mixed
        if half:
            lam_batch = torch.cat((lam_batch, torch.ones(num_elem)))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_pair_collate(self, output: torch.Tensor, batch: list):
        """
        This is the implementation for 'pair' mode
        :param output: the output tensor to fill
        :param batch: list of thr batch items
        :return: a tensor containing the lambda values used for the mixing (this vector can be used for
        mixing the labels as well)
        """
        batch_size = len(batch)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            mixed_i = batch[i][0]
            mixed_j = batch[j][0]
            assert 0 <= lam <= 1.0
            if lam < 1.0:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    patch_i = torch.clone(mixed_i[:, yl:yh, xl:xh])
                    mixed_i[:, yl:yh, xl:xh] = mixed_j[:, yl:yh, xl:xh]
                    mixed_j[:, yl:yh, xl:xh] = patch_i
                    lam_batch[i] = lam
                else:
                    mixed_temp = mixed_i.type(torch.float32) * lam + mixed_j.type(torch.float32) * (1 - lam)
                    mixed_j = mixed_j.type(torch.float32) * lam + mixed_i.type(torch.float32) * (1 - lam)
                    mixed_i = mixed_temp
                    torch.rint(mixed_j, out=mixed_j)
                    torch.rint(mixed_i, out=mixed_i)
            output[i] += mixed_i
            output[j] += mixed_j
        lam_batch = torch.cat((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch).unsqueeze(1)

    def _mix_batch_collate(self, output: torch.Tensor, batch: list):
        """
        This is the implementation for 'batch' mode
        :param output: the output tensor to fill
        :param batch: list of thr batch items
        :return: the lambda value used for the mixing
        """
        batch_size = len(batch)
        lam, use_cutmix = self._params_per_batch()
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(output.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
        for i in range(batch_size):
            j = batch_size - i - 1
            mixed = batch[i][0]
            if lam != 1.0:
                if use_cutmix:
                    mixed = torch.clone(mixed)  # don't want to modify the original while iterating
                    mixed[:, yl:yh, xl:xh] = batch[j][0][:, yl:yh, xl:xh]
                else:
                    mixed = mixed * lam + batch[j][0] * (1 - lam)
            output[i] += mixed
        return lam

    def __call__(self, batch, _=None):
        batch_size = len(batch)
        if batch_size % 2 != 0:
            raise IllegalDatasetParameterException("Batch size should be even when using this")
        half = "half" in self.mode
        if half:
            batch_size //= 2
        output = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        if self.mode == "elem" or self.mode == "half":
            lam = self._mix_elem_collate(output, batch, half=half)
        elif self.mode == "pair":
            lam = self._mix_pair_collate(output, batch)
        else:
            lam = self._mix_batch_collate(output, batch)
        target = torch.tensor([b[1] for b in batch], dtype=torch.int32)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device="cpu")
        target = target[:batch_size]

        return output, target
