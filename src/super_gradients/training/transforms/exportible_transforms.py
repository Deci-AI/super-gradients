import numbers
from typing import List

import torch.nn
from torch import Tensor
from torchvision.transforms.functional import pad, get_image_size


class ExportableCenterCrop(torch.nn.Module):
    def __init__(self, example_input: Tensor, crop_size: List[int]) -> Tensor:
        """Crops the given image at the center.
        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
        If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

        Args:
            example_input (PIL Image or Tensor): Image to be cropped.
            crop_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
                it is used for both directions.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        super(ExportableCenterCrop, self).__init__()
        self.no_crop_no_pad = False
        self.padding_ltrb = None

        if isinstance(crop_size, numbers.Number):
            crop_size = (int(crop_size), int(crop_size))
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 1:
            crop_size = (crop_size[0], crop_size[0])

        image_width, image_height = get_image_size(example_input)
        crop_height, crop_width = crop_size

        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            example_input = pad(example_input, padding_ltrb, fill=0)  # PIL uses fill value 0
            image_width, image_height = get_image_size(example_input)
            if crop_width == image_width and crop_height == image_height:
                self.no_crop_no_pad = True

        self.crop_top = int(round((image_height - crop_height) / 2.0))
        self.crop_left = int(round((image_width - crop_width) / 2.0))

        w, h = get_image_size(example_input)
        self.right = self.crop_left + crop_width
        self.bottom = self.crop_top + crop_height

        if self.crop_left < 0 or self.crop_top < 0 or self.right > w or self.bottom > h:
            self.padding_ltrb = [max(-self.crop_left, 0), max(-self.crop_top, 0), max(self.right - w, 0), max(self.bottom - h, 0)]

    def forward(self, img):
        if self.no_crop_no_pad:
            return img
        elif self.padding_ltrb is not None:
            return pad(img[..., max(self.crop_top, 0) : self.bottom, max(self.crop_left, 0) : self.right], self.padding_ltrb, fill=0)
        else:
            return img[..., self.crop_top : self.bottom, self.crop_left : self.right]
