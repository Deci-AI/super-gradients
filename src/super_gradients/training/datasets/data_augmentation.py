import numpy as np
import torch
from torchvision.transforms import RandomErasing

from super_gradients.common.object_names import Transforms
from super_gradients.common.registry.registry import register_transform


class DataAugmentation:
    @staticmethod
    def to_tensor():
        def _to_tensor(image):
            if len(image.shape) == 3:
                return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
            else:
                return torch.from_numpy(image[None, :, :].astype(np.float32))

        return _to_tensor

    @staticmethod
    def normalize(mean, std):
        mean = np.array(mean)
        std = np.array(std)

        def _normalize(image):
            image = np.asarray(image).astype(np.float32) / 255.0
            image = (image - mean) / std
            return image

        return _normalize

    @staticmethod
    def cutout(mask_size, p=1, cutout_inside=False, mask_color=(0, 0, 0)):
        mask_size_half = mask_size // 2
        offset = 1 if mask_size % 2 == 0 else 0

        def _cutout(image):
            image = np.asarray(image).copy()

            if np.random.random() > p:
                return image

            h, w = image.shape[:2]

            if cutout_inside:
                cxmin, cxmax = mask_size_half, w + offset - mask_size_half
                cymin, cymax = mask_size_half, h + offset - mask_size_half
            else:
                cxmin, cxmax = 0, w + offset
                cymin, cymax = 0, h + offset

            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
            xmin = cx - mask_size_half
            ymin = cy - mask_size_half
            xmax = xmin + mask_size
            ymax = ymin + mask_size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            image[ymin:ymax, xmin:xmax] = mask_color
            return image

        return _cutout


IMAGENET_PCA = {
    "eigval": torch.Tensor([0.2175, 0.0188, 0.0045]),
    "eigvec": torch.Tensor([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]),
}


@register_transform(Transforms.Lighting)
class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    Taken from fastai Imagenet training -
    https://github.com/fastai/imagenet-fast/blob/faa0f9dfc9e8e058ffd07a248724bf384f526fae/imagenet_nv/fastai_imagenet.py#L103
    To use:
        - training_params = {"imagenet_pca_aug": 0.1}
        - Default training_params arg is 0.0 ("don't use")
        - 0.1 is that default in the original paper
    """

    def __init__(self, alphastd, eigval=IMAGENET_PCA["eigval"], eigvec=IMAGENET_PCA["eigvec"]):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


@register_transform(Transforms.RandomErase)
class RandomErase(RandomErasing):
    """
    A simple class that translates the parameters supported in SuperGradient's code base
    """

    def __init__(self, probability: float, value: str):
        # value might be a string representing a float. First we try to convert to float and if fails,
        # pass it as-is to super
        try:
            value = float(value)
        except ValueError:
            pass
        super().__init__(p=probability, value=value)
