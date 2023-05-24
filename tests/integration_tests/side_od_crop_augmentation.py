import random

class RandomCrop():
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, min_crop_rel_x = 0.0, max_crop_rel_x = 0.5, side = None, p = 1.0):
        if side is not None:
            assert side in ["left", "right"], "side must be left or right"
        else:
            side = random.choice(["left", "right"])

        self.side = side
        self.min_crop_rel_x = min_crop_rel_x
        self.max_crop_rel_x = max_crop_rel_x
        self.p = p


    def apply(self, img, bboxes, **params):
        if random.random() > self.p:
            return img, bboxes
        
        crop_rel_x = random.uniform(self.min_crop_rel_x, self.max_crop_rel_x)

        if self.side == "left":
            return img[:, :int(img.shape[1]*self.crop_rel_x)]
        elif self.side == "right":
            return img[:, int(img.shape[1]*(1-self.crop_rel_x)):]
        
    