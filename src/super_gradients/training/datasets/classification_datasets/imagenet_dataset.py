import torchvision.datasets as torch_datasets
from torchvision.transforms import Compose

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.factories.list_factory import ListFactory


class ImageNetDataset(torch_datasets.ImageFolder):
    """ImageNetDataset dataset"""

    @resolve_param('transforms', factory=ListFactory(TransformsFactory()))
    def __init__(self, root: str, transforms: list = [], *args, **kwargs):
        super(ImageNetDataset, self).__init__(root, transform=Compose(transforms), *args, **kwargs)
