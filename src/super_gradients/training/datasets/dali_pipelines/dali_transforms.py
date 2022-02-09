try:
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI.")


class DaliTransform:
    def __init__(self, dali_transform_fn, **kwargs):
        self.dali_transform_fn = dali_transform_fn
        self.kwargs = kwargs

    def __call__(self, images):
        images = self.dali_transform_fn(images, **self.kwargs)
        return images


class DaliCompose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images


class DaliDecode(DaliTransform):
    """
    Plain image decoder wrapper for DALI
    """
    def __init__(self, **kwargs):
        super(DaliDecode, self).__init__(dali_transform_fn=fn.decoders.image, **kwargs)


class DaliDecodeRandomCrop(DaliTransform):
    """
    RandomCrop wrapper for DALI (decodes image first)
    """

    def __init__(self, **kwargs):
        super(DaliDecodeRandomCrop, self).__init__(dali_transform_fn=fn.decoders.image_random_crop, **kwargs)


class DaliResize(DaliTransform):
    """
    Resize wrapper for DALI
    """

    def __init__(self, **kwargs):
        super(DaliResize, self).__init__(dali_transform_fn=fn.resize, **kwargs)


class DaliCropMirrorNormalize(DaliTransform):
    """
    Crop, horizontal flip, normalize wrapper for DALI
    """

    def __init__(self, **kwargs):
        super(DaliCropMirrorNormalize, self).__init__(dali_transform_fn=fn.crop_mirror_normalize, **kwargs)

    def __call__(self, images):
        return super(DaliCropMirrorNormalize, self).__call__(images=images.gpu())
