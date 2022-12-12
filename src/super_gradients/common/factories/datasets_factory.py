from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.datasets.all_datasets import ALL_DATASETS


class DatasetsFactory(BaseFactory):
    def __init__(self):
        super().__init__(ALL_DATASETS)
