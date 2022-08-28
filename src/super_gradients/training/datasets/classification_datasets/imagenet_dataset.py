import torchvision.datasets as torch_datasets
import torchvision.transforms as torch_transforms

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory


class ImageNetDataset(torch_datasets.ImageFolder):
    """ImageNetDataset dataset"""

    @resolve_param('transform', factory=TransformsFactory())
    def __init__(self, root: str, transform: torch_transforms.Compose = None, *args, **kwargs):
        super(ImageNetDataset, self).__init__(root, transform, *args, **kwargs)
